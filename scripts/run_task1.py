from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.data_loader import PA4Data
from src.common.env import collect_run_metadata, save_json, set_seed
from src.common.metrics import paired_bootstrap_mean_diff, paired_bootstrap_ppl_diff
from src.common.timing import clear_cuda
from src.task1.eval_suite import (
    benchmark_dense_fp16_latency,
    evaluate_dev_nll,
    evaluate_hellaswag,
    evaluate_perplexity_windows,
    format_hellaswag_example,
    prepare_hellaswag_tokenized,
)
from src.task1.pruning import (
    allocate_nonuniform_sparsities,
    apply_nonuniform_wanda,
    apply_uniform_pruning,
    apply_wanda_2_4,
    collect_calibration_stats,
    compute_calibration_layer_errors,
    compute_pruning_sensitivities,
    restore_model_weights,
    snapshot_model_weights_cpu,
)
from src.task1.quantization import (
    apply_awq_with_bitmap,
    apply_q1_uniform_ptq,
    compute_diagonal_fisher_calibration,
    compute_quant_sensitivity_tables,
    solve_mixed_precision_knapsack,
)
from src.task1.storage import (
    compute_quant_storage_bytes,
    compute_sparse_storage_bytes,
    get_eligible_modules,
    verify_model_parameter_accounting,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ATML PA4 Task 1: Model Compression")
    p.add_argument("--data-root", type=str, default="atml-pa4-datasets")
    p.add_argument("--out-dir", type=str, default="results/task1")
    p.add_argument("--resume", action="store_true", help="Skip already-saved conditions")
    p.add_argument("--smoke-test", action="store_true", help="Fast ~2 min end-to-end verification of every code path")
    p.add_argument("--ppl-batch-size", type=int, default=4)
    p.add_argument("--hs-batch-size", type=int, default=32)
    p.add_argument("--timing-reps", type=int, default=5)
    return p.parse_args()


@torch.no_grad()
def capture_reference_block_outputs(
    model: torch.nn.Module,
    calib_rows: List[Dict[str, Any]],
    num_seqs: int = 8,
    device: str = "cuda",
) -> List[torch.Tensor]:
    """Capture unchanged FP16 block outputs on fixed calibration sequences (AWQ-folding invariant)."""
    model.eval()
    ids = torch.tensor([r["token_ids"] for r in calib_rows[:num_seqs]], dtype=torch.long, device=device)
    outputs: List[torch.Tensor] = [None] * len(model.model.layers)  # type: ignore[list-item]
    handles = []

    for idx, layer in enumerate(model.model.layers):
        def _make_hook(i: int):
            def _hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                outputs[i] = h.detach().to(torch.float16).cpu()
            return _hook
        handles.append(layer.register_forward_hook(_make_hook(idx)))

    _ = model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    return outputs


@torch.no_grad()
def evaluate_block_output_errors(
    model: torch.nn.Module,
    ref_block_outputs: List[torch.Tensor],
    calib_rows: List[Dict[str, Any]],
    num_seqs: int = 8,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Compute per-block relative L2 output distortion against unchanged FP16 reference."""
    model.eval()
    ids = torch.tensor([r["token_ids"] for r in calib_rows[:num_seqs]], dtype=torch.long, device=device)
    per_block: Dict[str, float] = {}
    handles = []

    for idx, layer in enumerate(model.model.layers):
        def _make_hook(i: int):
            def _hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                ref = ref_block_outputs[i].to(device=h.device, dtype=torch.float32)
                diff = h.detach().float() - ref
                rel = math.sqrt(
                    float(diff.pow(2).sum().item()) / max(float(ref.pow(2).sum().item()), 1e-12)
                )
                per_block[f"model.layers.{i}"] = rel
            return _hook
        handles.append(layer.register_forward_hook(_make_hook(idx)))

    _ = model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    vals = [per_block[f"model.layers.{i}"] for i in range(len(ref_block_outputs))]
    return {
        "mean_block_rel_l2": float(np.mean(vals)),
        "final_block_rel_l2": float(vals[-1]),
        "max_block_rel_l2": float(np.max(vals)),
        "per_block_rel_l2": per_block,
    }


def export_qualitative_hellaswag(
    hs_raw_rows: List[Dict[str, Any]],
    cond_dir: Path,
    cond_names: List[str],
    out_path: Path,
    num_cases: int = 8,
) -> None:
    """Export fixed matched HellaSwag cases across conditions (highlighting divergences)."""
    base_path = cond_dir / "baseline_fp16_details.json"
    if not base_path.exists():
        return
    details = {}
    for c in cond_names:
        p = cond_dir / f"{c}_details.json"
        if p.exists():
            details[c] = json.loads(p.read_text(encoding="utf-8"))["hellaswag_per_example"]

    n_common = min(len(v) for v in details.values())
    ranked = []
    for i in range(n_common):
        base_ok = details["baseline_fp16"][i]["correct_norm"]
        n_wrong = sum(1 - details[c][i]["correct_norm"] for c in details if c != "baseline_fp16")
        score = (10 if base_ok else 0) + (n_wrong if 0 < n_wrong < len(details) - 1 else 0)
        ranked.append((score, i))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    chosen_idxs = [idx for _, idx in ranked[:num_cases]]

    cases = []
    for idx in chosen_idxs:
        fmt = format_hellaswag_example(hs_raw_rows[idx])
        per_cond = {}
        for c, rows in details.items():
            r = rows[idx]
            per_cond[c] = {
                "pred_raw": r["pred_raw"],
                "pred_norm": r["pred_norm"],
                "correct_raw": r["correct_raw"],
                "correct_norm": r["correct_norm"],
                "raw_logliks": r["raw_logliks"],
                "norm_logliks": r["norm_logliks"],
            }
        cases.append({
            "diagnostic_index": idx,
            "order": fmt["order"],
            "source_row_index": fmt["source_row_index"],
            "query": fmt["query"],
            "choices": fmt["choices"],
            "gold_label": fmt["label"],
            "conditions": per_cond,
        })
    save_json(out_path, {"selection_rule": "baseline_correct_and_highest_cross_condition_disagreement_in_first_512", "cases": cases})


def main() -> None:
    args = parse_args()
    set_seed(6304)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_dir = out_dir / "conditions"
    cond_dir.mkdir(parents=True, exist_ok=True)

    smoke = args.smoke_test
    if smoke:
        print("=== [Task 1] SMOKE TEST MODE (exercising every condition & assertion on tiny slices) ===", flush=True)

    data = PA4Data(args.data_root)
    model_id, model_rev = data.model_spec("model_base")
    print(f"[Task 1] Loading {model_id} @ {model_rev} in FP16...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_rev, use_fast=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_rev,
        dtype=torch.float16,
        attn_implementation="sdpa",
    ).to("cuda").eval()

    acct = verify_model_parameter_accounting(model)
    eligible_mods = get_eligible_modules(model)
    eligible_shapes = {n: (m.out_features, m.in_features) for n, m in eligible_mods.items()}
    print(f"[Task 1] Parameter accounting: {json.dumps(acct)}", flush=True)

    # Load locked datasets (sliced cleanly in smoke-test mode)
    calib_rows = data.c4_calibration()
    dev_rows = data.c4_development()
    wt2_windows = data.wikitext2_test()
    c4_val_windows = data.c4_validation()
    hs_2000_raw = data.hellaswag_val(limit=2000)

    if smoke:
        calib_rows = calib_rows[:4]
        dev_rows = dev_rows[:4]
        wt2_windows = wt2_windows[:4]
        c4_val_windows = c4_val_windows[:4]
        hs_2000_raw = hs_2000_raw[:12]
        timing_reps, timing_warmup = 1, 1
        awq_grid = [0.0, 0.5, 1.0]
        q3_coarse_grid = [0.0, 0.5]
        p3_lambdas = (0.04, 0.08)
        fisher_seqs = 2
        block_err_seqs = 2
    else:
        timing_reps, timing_warmup = args.timing_reps, 3
        awq_grid = [round(i * 0.05, 2) for i in range(21)]
        q3_coarse_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        p3_lambdas = (0.04, 0.08, 0.12)
        fisher_seqs = 64
        block_err_seqs = 8

    hs_2000_prep = prepare_hellaswag_tokenized(hs_2000_raw, tokenizer)
    hs_512_prep = hs_2000_prep[:512]

    meta = collect_run_metadata(
        Path(args.data_root),
        extra={
            "task": "task1_model_compression",
            "smoke_test": smoke,
            "model_id": model_id,
            "model_revision": model_rev,
            "precision": "torch.float16",
            "attn_implementation": "sdpa",
            "seed": 6304,
            "parameter_accounting": acct,
        },
    )
    save_json(out_dir / "run_metadata.json", meta)
    orig_snapshot = snapshot_model_weights_cpu(model)

    ref_block_outputs = capture_reference_block_outputs(model, calib_rows, num_seqs=block_err_seqs)

    t0 = time.perf_counter()
    calib_stats = collect_calibration_stats(model, calib_rows, sample_tokens_per_seq=16, batch_size=4)
    calib_time_s = time.perf_counter() - t0
    print(f"[Task 1] Collected calibration stats ({calib_stats['total_tokens']} tokens) in {calib_time_s:.1f}s", flush=True)

    def eval_and_save(
        cond_name: str,
        full_hellaswag: bool,
        storage_info: Dict[str, Any],
        extra_info: Dict[str, Any],
        compression_time_s: float,
        measure_proj_err: bool = True,
        measure_block_err: bool = True,
    ) -> Dict[str, Any]:
        wt2 = evaluate_perplexity_windows(model, wt2_windows, batch_size=args.ppl_batch_size)
        c4v = evaluate_perplexity_windows(model, c4_val_windows, batch_size=args.ppl_batch_size)
        hs_prep = hs_2000_prep if full_hellaswag else hs_512_prep
        hs = evaluate_hellaswag(model, hs_prep, pad_token_id=pad_id, batch_size=args.hs_batch_size)
        latency = benchmark_dense_fp16_latency(
            model, vocab_size=model.config.vocab_size, reps=timing_reps, warmup=timing_warmup,
        )
        layer_err = (
            compute_calibration_layer_errors(orig_snapshot, model, calib_stats["x_samples"])
            if measure_proj_err else None
        )
        block_err = (
            evaluate_block_output_errors(model, ref_block_outputs, calib_rows, num_seqs=block_err_seqs)
            if measure_block_err else None
        )

        save_json(
            cond_dir / f"{cond_name}_details.json",
            {
                "wt2_window_nll_sums": wt2.pop("window_nll_sums"),
                "wt2_window_scored_tokens": wt2.pop("window_scored_tokens"),
                "c4_window_nll_sums": c4v.pop("window_nll_sums"),
                "c4_window_scored_tokens": c4v.pop("window_scored_tokens"),
                "hellaswag_per_example": hs.pop("per_example"),
                "layer_calibration_error": layer_err,
                "block_calibration_error": block_err,
            },
        )
        summary = {
            "condition": cond_name,
            "full_hellaswag_2000": full_hellaswag,
            "compression_time_s": compression_time_s,
            "shared_calibration_time_s": calib_time_s,
            "wikitext2": wt2,
            "c4_validation": c4v,
            "hellaswag": hs,
            "storage": storage_info,
            "latency": latency,
            "calib_error_summary": (
                {k: v for k, v in layer_err.items() if k != "per_projection_rel_l2"}
                if layer_err else None
            ),
            "block_calib_error_summary": (
                {k: v for k, v in block_err.items() if k != "per_block_rel_l2"}
                if block_err else None
            ),
            **extra_info,
        }
        save_json(cond_dir / f"{cond_name}.json", summary)
        print(
            f"  -> {cond_name:24s} | WT2 PPL={wt2['ppl']:.3f} | C4 PPL={c4v['ppl']:.3f} | "
            f"HS({hs['num_examples']}) acc={hs['acc']*100:.2f}% acc_norm={hs['acc_norm']*100:.2f}% | "
            f"packed={storage_info['total_packed_bytes']/1e6:.1f} MB",
            flush=True,
        )
        return summary

    def already_done(cond_name: str) -> bool:
        return (
            not smoke
            and args.resume
            and (cond_dir / f"{cond_name}.json").exists()
            and (cond_dir / f"{cond_name}_details.json").exists()
        )

    def load_done(cond_name: str) -> Dict[str, Any]:
        return json.loads((cond_dir / f"{cond_name}.json").read_text(encoding="utf-8"))

    all_summaries: Dict[str, Dict[str, Any]] = {}

    # 1. FP16 Baseline
    if not already_done("baseline_fp16"):
        restore_model_weights(model, orig_snapshot)
        st = compute_sparse_storage_bytes(model, acct)
        st["total_packed_bytes"] = acct["total_unique_fp16_bytes"]
        st["nominal_bits_per_eligible_weight"] = 16.0
        st["compression_ratio_vs_fp16"] = 1.0
        all_summaries["baseline_fp16"] = eval_and_save(
            "baseline_fp16", True, st, {"family": "baseline"}, 0.0,
            measure_proj_err=False, measure_block_err=False,
        )
    else:
        all_summaries["baseline_fp16"] = load_done("baseline_fp16")

    # 2. P1: Uniform Magnitude Pruning (25%, 50%, 75%)
    for s in (0.25, 0.50, 0.75):
        cname = f"P1_mag_s{int(s*100)}"
        if not already_done(cname):
            restore_model_weights(model, orig_snapshot)
            t_c = time.perf_counter()
            info = apply_uniform_pruning(model, sparsity=s, method="magnitude")
            dt = time.perf_counter() - t_c
            st = compute_sparse_storage_bytes(model, acct)
            all_summaries[cname] = eval_and_save(
                cname, full_hellaswag=(s == 0.50), storage_info=st,
                extra_info={"family": "P1_magnitude", "target_sparsity": s, **info},
                compression_time_s=dt,
            )
        else:
            all_summaries[cname] = load_done(cname)

    # 3. P2: Uniform Wanda Pruning (25%, 50%, 75%) + 2:4 Diagnostic
    for s in (0.25, 0.50, 0.75):
        cname = f"P2_wanda_s{int(s*100)}"
        if not already_done(cname):
            restore_model_weights(model, orig_snapshot)
            t_c = time.perf_counter()
            info = apply_uniform_pruning(model, sparsity=s, method="wanda", x_l2_norm=calib_stats["x_l2_norm"])
            dt = time.perf_counter() - t_c
            st = compute_sparse_storage_bytes(model, acct)
            all_summaries[cname] = eval_and_save(
                cname, full_hellaswag=(s == 0.50), storage_info=st,
                extra_info={"family": "P2_wanda", "target_sparsity": s, **info},
                compression_time_s=dt,
            )
        else:
            all_summaries[cname] = load_done(cname)

    if not already_done("P2_wanda_2to4_s50"):
        restore_model_weights(model, orig_snapshot)
        t_c = time.perf_counter()
        info = apply_wanda_2_4(model, x_l2_norm=calib_stats["x_l2_norm"])
        dt = time.perf_counter() - t_c
        st = compute_sparse_storage_bytes(model, acct)
        all_summaries["P2_wanda_2to4_s50"] = eval_and_save(
            "P2_wanda_2to4_s50", full_hellaswag=False, storage_info=st,
            extra_info={"family": "P2_wanda_2to4", "target_sparsity": 0.50, **info},
            compression_time_s=dt,
        )
    else:
        all_summaries["P2_wanda_2to4_s50"] = load_done("P2_wanda_2to4_s50")

    # 4. P3: Nonuniform Wanda Pruning (50%, 75%)
    for s in (0.50, 0.75):
        cname = f"P3_nonuniform_wanda_s{int(s*100)}"
        if not already_done(cname):
            t_probe = time.perf_counter()
            sens_dict = compute_pruning_sensitivities(
                model, orig_snapshot, calib_stats["x_l2_norm"], calib_stats["x_samples"], target_sparsity=s,
            )
            probe_time = time.perf_counter() - t_probe

            candidates = []
            best_cand = None
            for probe_name in ("owl_outlier_ratio", "calib_distortion"):
                for lam in p3_lambdas:
                    restore_model_weights(model, orig_snapshot)
                    alloc_map, row_zeros, residual = allocate_nonuniform_sparsities(
                        eligible_shapes, sens_dict[probe_name], target_sparsity=s, lam=lam,
                    )
                    apply_nonuniform_wanda(model, calib_stats["x_l2_norm"], row_zeros, residual)
                    dev_nll = evaluate_dev_nll(model, dev_rows, batch_size=args.ppl_batch_size)
                    cand_rec = {
                        "probe": probe_name,
                        "lambda": lam,
                        "dev_c4_nll": dev_nll,
                        "dev_c4_ppl": float(math.exp(dev_nll)) if dev_nll < 80.0 else float("inf"),
                        "min_s": float(min(alloc_map.values())),
                        "max_s": float(max(alloc_map.values())),
                    }
                    candidates.append(cand_rec)
                    print(
                        f"  [P3 s={int(s*100)} dev] probe={probe_name:18s} lam={lam:.2f} -> dev_ppl={cand_rec['dev_c4_ppl']:.3f}",
                        flush=True,
                    )
                    if best_cand is None or dev_nll < best_cand["dev_c4_nll"]:
                        best_cand = {**cand_rec, "row_zeros": row_zeros, "residual": residual}

            assert best_cand is not None
            save_json(
                out_dir / f"p3_selection_s{int(s*100)}.json",
                {
                    "probe_time_s": probe_time,
                    "sensitivities": sens_dict,
                    "dev_candidates": candidates,
                    "selected": {k: v for k, v in best_cand.items() if k != "row_zeros"},
                },
            )

            restore_model_weights(model, orig_snapshot)
            t_c = time.perf_counter()
            apply_info = apply_nonuniform_wanda(
                model, calib_stats["x_l2_norm"], best_cand["row_zeros"], best_cand["residual"],
            )
            dt = probe_time + (time.perf_counter() - t_c)
            st = compute_sparse_storage_bytes(model, acct)
            p2_zeros = all_summaries[f"P2_wanda_s{int(s*100)}"]["storage"]["eligible_zeros"]
            assert st["eligible_zeros"] == p2_zeros, (
                f"Budget mismatch! P3 zeros={st['eligible_zeros']} vs P2={p2_zeros}"
            )
            all_summaries[cname] = eval_and_save(
                cname, full_hellaswag=(s == 0.50), storage_info=st,
                extra_info={
                    "family": "P3_nonuniform_wanda",
                    "target_sparsity": s,
                    "selected_probe": best_cand["probe"],
                    "selected_lambda": best_cand["lambda"],
                    "per_projection_sparsity": apply_info["per_projection_sparsity"],
                    "matched_zeros_exact": True,
                },
                compression_time_s=dt,
            )
        else:
            all_summaries[cname] = load_done(cname)

    # 5. Q1: Uniform Asymmetric Group-128 PTQ (8, 4, 3 bits)
    for b in (8, 4, 3):
        cname = f"Q1_uniform_int{b}"
        if not already_done(cname):
            restore_model_weights(model, orig_snapshot)
            t_c = time.perf_counter()
            bit_map = apply_q1_uniform_ptq(model, bits=b, group_size=128)
            dt = time.perf_counter() - t_c
            st = compute_quant_storage_bytes(eligible_shapes, bit_map, acct, group_size=128)
            all_summaries[cname] = eval_and_save(
                cname, full_hellaswag=(b in (4, 3)), storage_info=st,
                extra_info={"family": "Q1_uniform_ptq", "bits": b, "group_size": 128},
                compression_time_s=dt,
            )
        else:
            all_summaries[cname] = load_done(cname)

    # 6. Q2: Activation-Aware Weight Quantization (AWQ) (8, 4, 3 bits)
    verify_ids = torch.tensor([calib_rows[0]["token_ids"]], dtype=torch.long, device="cuda")
    q2_unfolded_counts: Dict[str, int] = {
        n: eligible_shapes[n][1] for n in eligible_shapes if n.endswith("self_attn.o_proj")
    }
    for b in (8, 4, 3):
        cname = f"Q2_awq_int{b}"
        if not already_done(cname):
            restore_model_weights(model, orig_snapshot)
            t_c = time.perf_counter()
            bit_map = {n: b for n in eligible_shapes}
            awq_info = apply_awq_with_bitmap(
                model, bit_map, calib_stats["x_abs_mean"], calib_stats["x_samples"],
                group_size=128, alpha_grid=awq_grid, verify_pre_quant_calib_ids=verify_ids,
            )
            dt = time.perf_counter() - t_c
            q2_unfolded_counts = awq_info["unfolded_scale_counts"]
            st = compute_quant_storage_bytes(
                eligible_shapes, bit_map, acct, group_size=128,
                unfolded_scale_counts=q2_unfolded_counts,
            )
            all_summaries[cname] = eval_and_save(
                cname, full_hellaswag=(b in (4, 3)), storage_info=st,
                extra_info={
                    "family": "Q2_awq",
                    "bits": b,
                    "group_size": 128,
                    "awq_pre_quant_equivalence": awq_info["pre_quant_equivalence"],
                    "awq_alpha_grid": awq_info["alpha_grid"],
                    "awq_chosen_alphas": awq_info["chosen_alphas"],
                },
                compression_time_s=dt,
                measure_proj_err=False,
                measure_block_err=True,
            )
        else:
            all_summaries[cname] = load_done(cname)

    # 7. Q3: Mixed Precision AWQ (Matched to Q2 INT4 and Q2 INT3 Packed Bytes)
    need_q3 = any(not already_done(f"Q3_mixed_awq_budget_int{b}") for b in (4, 3))
    if need_q3:
        sens_cache_path = out_dir / "q3_sensitivity_tables.json"
        if not smoke and args.resume and sens_cache_path.exists():
            raw_tables = json.loads(sens_cache_path.read_text(encoding="utf-8"))
            sens_tables = {
                probe: {proj: {int(bk): float(bv) for bk, bv in bdict.items()} for proj, bdict in pdict.items()}
                for probe, pdict in raw_tables.items()
            }
            q3_probe_time = 0.0
            print("[Task 1] Loaded cached Q3 sensitivity tables.", flush=True)
        else:
            restore_model_weights(model, orig_snapshot)
            clear_cuda()
            t_q3_probe = time.perf_counter()
            print("[Task 1] Computing diagonal Fisher & AWQ distortion tables for Q3...", flush=True)
            fisher_diag = compute_diagonal_fisher_calibration(model, calib_rows, num_seqs=fisher_seqs)
            sens_tables = compute_quant_sensitivity_tables(
                model, orig_snapshot, calib_stats["x_abs_mean"], calib_stats["x_samples"],
                fisher_diag, coarse_grid=q3_coarse_grid,
            )
            q3_probe_time = time.perf_counter() - t_q3_probe
            save_json(sens_cache_path, sens_tables)
            del fisher_diag
            clear_cuda()

        for target_b in (4, 3):
            cname = f"Q3_mixed_awq_budget_int{target_b}"
            if already_done(cname):
                all_summaries[cname] = load_done(cname)
                continue

            q2_ref_storage = all_summaries[f"Q2_awq_int{target_b}"]["storage"]
            target_elig_bytes = int(q2_ref_storage["eligible_packed_bytes"])
            target_total_bytes = int(q2_ref_storage["total_packed_bytes"])

            dev_cands = []
            best_q3 = None
            for probe_name in ("awq_output_mse", "awq_fisher_loss"):
                bit_map = solve_mixed_precision_knapsack(
                    eligible_shapes,
                    sens_tables[probe_name],
                    target_eligible_bytes=target_elig_bytes,
                    unfolded_scale_counts=q2_unfolded_counts,
                    allowed_bits=(2, 3, 4, 8),
                    group_size=128,
                )
                restore_model_weights(model, orig_snapshot)
                apply_awq_with_bitmap(
                    model, bit_map, calib_stats["x_abs_mean"], calib_stats["x_samples"],
                    group_size=128, alpha_grid=awq_grid,
                )
                dev_nll = evaluate_dev_nll(model, dev_rows, batch_size=args.ppl_batch_size)
                st_cand = compute_quant_storage_bytes(
                    eligible_shapes, bit_map, acct, group_size=128, unfolded_scale_counts=q2_unfolded_counts,
                )
                assert st_cand["total_packed_bytes"] <= target_total_bytes
                bit_hist = {str(b): sum(1 for v in bit_map.values() if v == b) for b in (2, 3, 4, 8)}
                rec = {
                    "probe": probe_name,
                    "dev_c4_nll": dev_nll,
                    "dev_c4_ppl": float(math.exp(dev_nll)) if dev_nll < 80.0 else float("inf"),
                    "bit_histogram": bit_hist,
                    "total_packed_bytes": st_cand["total_packed_bytes"],
                    "budget_packed_bytes": target_total_bytes,
                }
                dev_cands.append(rec)
                print(
                    f"  [Q3 int{target_b} dev] probe={probe_name:16s} -> dev_ppl={rec['dev_c4_ppl']:.3f} | hist={bit_hist}",
                    flush=True,
                )
                if best_q3 is None or dev_nll < best_q3["dev_c4_nll"]:
                    best_q3 = {**rec, "bit_map": bit_map}

            assert best_q3 is not None
            save_json(
                out_dir / f"q3_selection_int{target_b}.json",
                {
                    "probe_time_s": q3_probe_time,
                    "dev_candidates": dev_cands,
                    "selected": {k: v for k, v in best_q3.items() if k != "bit_map"},
                },
            )

            restore_model_weights(model, orig_snapshot)
            t_c = time.perf_counter()
            awq_info = apply_awq_with_bitmap(
                model, best_q3["bit_map"], calib_stats["x_abs_mean"], calib_stats["x_samples"],
                group_size=128, alpha_grid=awq_grid, verify_pre_quant_calib_ids=verify_ids,
            )
            dt = q3_probe_time + (time.perf_counter() - t_c)
            st = compute_quant_storage_bytes(
                eligible_shapes, best_q3["bit_map"], acct, group_size=128,
                unfolded_scale_counts=awq_info["unfolded_scale_counts"],
            )
            assert st["total_packed_bytes"] <= target_total_bytes, (
                f"Q3 exceeded Q2 INT{target_b} byte budget: {st['total_packed_bytes']} > {target_total_bytes}"
            )
            all_summaries[cname] = eval_and_save(
                cname, full_hellaswag=True, storage_info=st,
                extra_info={
                    "family": "Q3_mixed_precision_awq",
                    "matched_budget_of": f"Q2_awq_int{target_b}",
                    "budget_packed_bytes": target_total_bytes,
                    "selected_probe": best_q3["probe"],
                    "bit_histogram": best_q3["bit_histogram"],
                    "bit_map": best_q3["bit_map"],
                    "awq_pre_quant_equivalence": awq_info["pre_quant_equivalence"],
                },
                compression_time_s=dt,
                measure_proj_err=False,
                measure_block_err=True,
            )
    else:
        for target_b in (4, 3):
            cname = f"Q3_mixed_awq_budget_int{target_b}"
            all_summaries[cname] = load_done(cname)

    # 8. Paired 95% Bootstrap CIs for P3-P2 and Q3-Q2 + Qualitative Export
    pairs = [
        ("P3_nonuniform_wanda_s50", "P2_wanda_s50"),
        ("P3_nonuniform_wanda_s75", "P2_wanda_s75"),
        ("Q3_mixed_awq_budget_int4", "Q2_awq_int4"),
        ("Q3_mixed_awq_budget_int3", "Q2_awq_int3"),
        ("P2_wanda_2to4_s50", "P2_wanda_s50"),
    ]
    boot_results: Dict[str, Any] = {}
    for cond_a, cond_b in pairs:
        da = json.loads((cond_dir / f"{cond_a}_details.json").read_text(encoding="utf-8"))
        db = json.loads((cond_dir / f"{cond_b}_details.json").read_text(encoding="utf-8"))
        n_hs = min(len(da["hellaswag_per_example"]), len(db["hellaswag_per_example"]))
        boot_results[f"{cond_a}_minus_{cond_b}"] = {
            "wikitext2": paired_bootstrap_ppl_diff(
                da["wt2_window_nll_sums"], db["wt2_window_nll_sums"], da["wt2_window_scored_tokens"],
            ),
            "c4_validation": paired_bootstrap_ppl_diff(
                da["c4_window_nll_sums"], db["c4_window_nll_sums"], da["c4_window_scored_tokens"],
            ),
            "hellaswag_acc": paired_bootstrap_mean_diff(
                [x["correct_raw"] for x in da["hellaswag_per_example"][:n_hs]],
                [x["correct_raw"] for x in db["hellaswag_per_example"][:n_hs]],
            ),
            "hellaswag_acc_norm": paired_bootstrap_mean_diff(
                [x["correct_norm"] for x in da["hellaswag_per_example"][:n_hs]],
                [x["correct_norm"] for x in db["hellaswag_per_example"][:n_hs]],
            ),
        }

    export_qualitative_hellaswag(
        hs_2000_raw, cond_dir, list(all_summaries.keys()), out_dir / "qualitative_hellaswag_cases.json",
    )
    save_json(out_dir / "paired_bootstrap_cis.json", boot_results)
    save_json(out_dir / "summary_all_conditions.json", all_summaries)
    tag = "SMOKE TEST PASSED" if smoke else "COMPLETE"
    print(f"\n[Task 1] {tag}! Saved summary_all_conditions.json, paired_bootstrap_cis.json, and qualitative_hellaswag_cases.json.", flush=True)


if __name__ == "__main__":
    main()
