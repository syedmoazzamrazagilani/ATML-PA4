from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.common.timing import clear_cuda, measure_cuda_fn


def preprocess_hellaswag_text(text: str) -> str:
    """Standard lm-evaluation-harness HellaSwag text preprocessing."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text.strip()


def format_hellaswag_example(row: Dict[str, Any]) -> Dict[str, Any]:
    ex = row["example"]
    ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
    query = preprocess_hellaswag_text(ex["activity_label"] + ": " + ctx)
    choices = [preprocess_hellaswag_text(e) for e in ex["endings"]]
    return {
        "order": int(row["order"]),
        "source_row_index": int(row["source_row_index"]),
        "query": query,
        "choices": choices,
        "label": int(ex["label"]),
    }


@torch.inference_mode()
def evaluate_perplexity_windows(
    model: torch.nn.Module,
    windows: List[Dict[str, Any]],
    batch_size: int = 4,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate nonoverlapping fixed windows (e.g., 50,000 scored tokens on WT2 or C4 val).
    Excludes first token of each window from the scored-token denominator.
    Saves per-window NLL sums and scored token counts for exact paired bootstrap.
    """
    model.eval()
    window_nll_sums: List[float] = []
    window_scored_tokens: List[int] = []

    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        seqs = [w["token_ids"] for w in batch]
        max_len = max(len(s) for s in seqs)
        pad_id = 0

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            attn_mask[i, : len(s)] = 1

        logits = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]

        loss_per_tok = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view(len(batch), -1)

        for i, w in enumerate(batch):
            n_scored = int(w["scored_tokens"])
            assert n_scored == len(w["token_ids"]) - 1
            nll_sum = float(loss_per_tok[i, :n_scored].sum().item())
            window_nll_sums.append(nll_sum)
            window_scored_tokens.append(n_scored)

    total_tokens = int(sum(window_scored_tokens))
    total_nll = float(sum(window_nll_sums))
    mean_nll = total_nll / total_tokens
    ppl = float(math.exp(mean_nll)) if mean_nll < 80.0 else float("inf")
    return {
        "scored_tokens": total_tokens,
        "num_windows": len(windows),
        "mean_nll": mean_nll,
        "ppl": ppl,
        "window_nll_sums": window_nll_sums,
        "window_scored_tokens": window_scored_tokens,
    }


@torch.inference_mode()
def evaluate_dev_nll(
    model: torch.nn.Module,
    dev_rows: List[Dict[str, Any]],
    batch_size: int = 4,
    device: str = "cuda",
) -> float:
    """Fast mean NLL on the 32-sequence C4 development manifest for P3/Q3 selection."""
    windows = [
        {"token_ids": r["token_ids"], "scored_tokens": len(r["token_ids"]) - 1}
        for r in dev_rows
    ]
    res = evaluate_perplexity_windows(model, windows, batch_size=batch_size, device=device)
    return float(res["mean_nll"])


def prepare_hellaswag_tokenized(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    """Pre-tokenize HellaSwag once so every model sees identical inputs and target spans."""
    prepared: List[Dict[str, Any]] = []
    for row in rows:
        fmt = format_hellaswag_example(row)
        ctx_ids = tokenizer.encode(fmt["query"], add_special_tokens=False)
        cand_items = []
        for choice in fmt["choices"]:
            full_ids = tokenizer.encode(fmt["query"] + " " + choice, add_special_tokens=False)
            if len(full_ids) <= len(ctx_ids) or full_ids[: len(ctx_ids)] != ctx_ids:
                cont_ids = tokenizer.encode(" " + choice, add_special_tokens=False)
                full_ids = ctx_ids + cont_ids
            start_idx = len(ctx_ids)
            cand_items.append({
                "full_ids": full_ids,
                "start_idx": start_idx,
                "char_len": max(len(choice), 1),
            })
        prepared.append({
            "order": fmt["order"],
            "source_row_index": fmt["source_row_index"],
            "label": fmt["label"],
            "candidates": cand_items,
        })
    return prepared


@torch.inference_mode()
def evaluate_hellaswag(
    model: torch.nn.Module,
    prepared_examples: List[Dict[str, Any]],
    pad_token_id: int,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Zero-shot HellaSwag conditional log-likelihood evaluation.
    Reports raw accuracy (acc) and character-length-normalized accuracy (acc_norm).
    """
    model.eval()
    flat_items = []
    for ex_idx, ex in enumerate(prepared_examples):
        for c_idx, cand in enumerate(ex["candidates"]):
            flat_items.append((ex_idx, c_idx, cand["full_ids"], cand["start_idx"], cand["char_len"]))

    # Sort by length for minimal padding waste
    order = sorted(range(len(flat_items)), key=lambda i: len(flat_items[i][2]))
    raw_scores = [[0.0] * 4 for _ in range(len(prepared_examples))]
    norm_scores = [[0.0] * 4 for _ in range(len(prepared_examples))]

    for b_start in range(0, len(order), batch_size):
        idxs = order[b_start : b_start + batch_size]
        max_len = max(len(flat_items[i][2]) for i in idxs)
        input_ids = torch.full((len(idxs), max_len), pad_token_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((len(idxs), max_len), dtype=torch.long, device=device)
        for row_i, item_i in enumerate(idxs):
            seq = flat_items[item_i][2]
            input_ids[row_i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attn_mask[row_i, : len(seq)] = 1

        logits = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits
        log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = input_ids[:, 1:]
        tok_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        for row_i, item_i in enumerate(idxs):
            ex_idx, c_idx, seq, start_idx, char_len = flat_items[item_i]
            # Token at position t is predicted by logits at t-1 -> index t-1 in tok_logp
            ll = float(tok_logp[row_i, start_idx - 1 : len(seq) - 1].sum().item())
            raw_scores[ex_idx][c_idx] = ll
            norm_scores[ex_idx][c_idx] = ll / float(char_len)

    correct_raw = []
    correct_norm = []
    per_example = []
    for ex_idx, ex in enumerate(prepared_examples):
        r = raw_scores[ex_idx]
        n = norm_scores[ex_idx]
        pred_r = int(max(range(4), key=lambda c: r[c]))
        pred_n = int(max(range(4), key=lambda c: n[c]))
        cr = int(pred_r == ex["label"])
        cn = int(pred_n == ex["label"])
        correct_raw.append(cr)
        correct_norm.append(cn)
        per_example.append({
            "order": ex["order"],
            "source_row_index": ex["source_row_index"],
            "label": ex["label"],
            "pred_raw": pred_r,
            "pred_norm": pred_n,
            "correct_raw": cr,
            "correct_norm": cn,
            "raw_logliks": [round(x, 4) for x in r],
            "norm_logliks": [round(x, 6) for x in n],
        })

    return {
        "num_examples": len(prepared_examples),
        "acc": float(sum(correct_raw) / len(correct_raw)),
        "acc_norm": float(sum(correct_norm) / len(correct_norm)),
        "prompt_convention": "lm_eval_hellaswag_preprocess_activity_colon_ctx_space_ending",
        "norm_convention": "character_length_of_preprocessed_ending",
        "per_example": per_example,
    }


@torch.inference_mode()
def benchmark_dense_fp16_latency(
    model: torch.nn.Module,
    vocab_size: int,
    reps: int = 5,
    warmup: int = 3,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Measure dense FP16 execution cost:
      - Prefill latency for 512 and 2,048 token prompts
      - 64-token greedy decode latency (from a 512-token prompt, excluding prefill)
      - Peak allocated GPU memory
    Explicitly labeled as dense FP16 execution (no custom sparse/low-bit kernel).
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(6304)
    ids_512 = torch.randint(10, min(vocab_size, 50000), (1, 512), generator=gen, device=device)
    ids_2048 = torch.randint(10, min(vocab_size, 50000), (1, 2048), generator=gen, device=device)

    clear_cuda()
    prefill_512 = measure_cuda_fn(
        lambda: model(input_ids=ids_512, use_cache=True),
        warmup=warmup, reps=reps, reset_peak_per_rep=True,
    )
    clear_cuda()
    prefill_2048 = measure_cuda_fn(
        lambda: model(input_ids=ids_2048, use_cache=True),
        warmup=warmup, reps=reps, reset_peak_per_rep=True,
    )

    def decode_64_from_512():
        out = model(input_ids=ids_512, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Time only the 64 autoregressive decode steps separately via inner helper below
        for _ in range(63):
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        return next_tok

    # Measure pure 64-step decode (prefill excluded from timed window)
    def run_decode_only() -> Dict[str, Any]:
        is_cuda = torch.cuda.is_available()
        times_ms = []
        peak_alloc = 0
        peak_res = 0
        for r in range(warmup + reps):
            out = model(input_ids=ids_512, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            if is_cuda:
                torch.cuda.synchronize()
                if r >= warmup:
                    torch.cuda.reset_peak_memory_stats()
                s_evt = torch.cuda.Event(enable_timing=True)
                e_evt = torch.cuda.Event(enable_timing=True)
                s_evt.record()
            for _ in range(64):
                out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            if is_cuda:
                e_evt.record()
                torch.cuda.synchronize()
                if r >= warmup:
                    times_ms.append(float(s_evt.elapsed_time(e_evt)))
                    peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated()))
                    peak_res = max(peak_res, int(torch.cuda.max_memory_reserved()))
        import numpy as np
        arr = np.asarray(times_ms, dtype=np.float64)
        return {
            "unit": "ms",
            "warmup": warmup,
            "reps": reps,
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "median_ms": float(np.median(arr)),
            "tokens_per_sec": float(64.0 / (arr.mean() / 1000.0)),
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_res,
        }

    clear_cuda()
    decode_64 = run_decode_only()
    return {
        "execution_backend": "dense_fp16_sdpa_no_custom_sparse_or_lowbit_kernel",
        "prefill_512": prefill_512,
        "prefill_2048": prefill_2048,
        "decode_64_from_512": decode_64,
        "peak_allocated_bytes_overall": max(
            prefill_512["peak_allocated_bytes"],
            prefill_2048["peak_allocated_bytes"],
            decode_64["peak_allocated_bytes"],
        ),
    }
