from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.common.env import collect_run_metadata, save_json, set_seed
from src.common.timing import clear_cuda, measure_cuda_fn


def dense_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A1: Explicit QK^T / sqrt(d_h), causal mask, stable softmax, multiplication by V."""
    L = q.shape[-2]
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal_mask = torch.triu(
        torch.ones((L, L), device=q.device, dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    # Subtract max for stability; compute softmax in FP32 for numerical fidelity, cast back
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


def tiled_online_softmax_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tile_q: int = 128,
    tile_k: int = 128,
) -> torch.Tensor:
    """
    A2: Exact blockwise online-softmax causal attention without materializing the full L x L score matrix.
    Note: When run with autograd enabled in pure Python, PyTorch saves per-tile intermediates for backward;
    this is a numerical reference, not a fused FlashAttention backward kernel.
    """
    B, H, L, d = q.shape
    scale = 1.0 / math.sqrt(d)
    out_blocks: List[torch.Tensor] = []

    for q_start in range(0, L, tile_q):
        q_end = min(q_start + tile_q, L)
        q_blk = q[:, :, q_start:q_end, :]  # [B, H, bq, d]
        bq = q_end - q_start

        m = torch.full((B, H, bq, 1), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, bq, 1), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, bq, d), device=q.device, dtype=torch.float32)

        q_pos = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)  # [bq, 1]

        # Only iterate over KV tiles that contain at least one causal key (k_start < q_end)
        for k_start in range(0, q_end, tile_k):
            k_end = min(k_start + tile_k, L)
            k_blk = k[:, :, k_start:k_end, :]
            v_blk = v[:, :, k_start:k_end, :]

            s_tile = torch.matmul(q_blk, k_blk.transpose(-2, -1)).float() * scale
            if k_end > q_start:
                k_pos = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)  # [1, bk]
                mask = k_pos > q_pos
                s_tile = s_tile.masked_fill(mask, float("-inf"))

            tile_max = s_tile.amax(dim=-1, keepdim=True)
            m_new = torch.maximum(m, tile_max)

            # Safe exponential guarding against -inf - (-inf) = NaN on all-masked rows
            alpha = torch.where(torch.isneginf(m_new), torch.zeros_like(m), torch.exp(m - m_new))
            p_tile = torch.where(
                torch.isneginf(s_tile),
                torch.zeros_like(s_tile),
                torch.exp(s_tile - m_new),
            )

            l = alpha * l + p_tile.sum(dim=-1, keepdim=True)
            o = alpha * o + torch.matmul(p_tile, v_blk.float())
            m = m_new

        out_blk = (o / l.clamp_min(1e-12)).to(q.dtype)
        out_blocks.append(out_blk)

    return torch.cat(out_blocks, dim=-2)


def sdpa_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: Optional[SDPBackend] = None,
) -> torch.Tensor:
    """A3: PyTorch scaled_dot_product_attention with optional forced backend."""
    if backend is None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)


def run_causality_and_edge_checks(device: str = "cuda") -> Dict[str, Any]:
    """Verify strict causality by perturbing future K/V and check extreme-logit numerical stability."""
    set_seed(6304)
    B, H, L, d = 2, 4, 256, 64
    q = torch.randn(B, H, L, d, device=device, dtype=torch.float32)
    k = torch.randn(B, H, L, d, device=device, dtype=torch.float32)
    v = torch.randn(B, H, L, d, device=device, dtype=torch.float32)

    k_pert = k.clone()
    v_pert = v.clone()
    t_split = 128
    k_pert[:, :, t_split:, :] += 10.0
    v_pert[:, :, t_split:, :] -= 10.0

    causality = {}
    for name, fn in [
        ("dense", dense_causal_attention),
        ("tiled_64", lambda a, b, c: tiled_online_softmax_causal_attention(a, b, c, 64, 64)),
        ("sdpa_default", lambda a, b, c: sdpa_causal_attention(a, b, c, None)),
    ]:
        y1 = fn(q, k, v)
        y2 = fn(q, k_pert, v_pert)
        prefix_diff = float((y1[:, :, :t_split, :] - y2[:, :, :t_split, :]).abs().max().item())
        suffix_diff = float((y1[:, :, t_split:, :] - y2[:, :, t_split:, :]).abs().max().item())
        causality[name] = {
            "prefix_max_abs_diff": prefix_diff,
            "suffix_max_abs_diff": suffix_diff,
            "strictly_causal": prefix_diff == 0.0 and suffix_diff > 0.0,
        }

    # Tile-size agreement check across 64, 128, 256 in FP32 and FP16
    tile_checks = {}
    for dtype_name, dt in [("fp32", torch.float32), ("fp16", torch.float16)]:
        ref = dense_causal_attention(q.to(dt), k.to(dt), v.to(dt)).float()
        tile_checks[dtype_name] = {}
        for ts in (64, 128, 256):
            yt = tiled_online_softmax_causal_attention(q.to(dt), k.to(dt), v.to(dt), ts, ts).float()
            diff = (yt - ref).abs()
            tile_checks[dtype_name][f"tile_{ts}"] = {
                "max_abs_err": float(diff.max().item()),
                "mean_abs_err": float(diff.mean().item()),
            }

    # Extreme-score stability check (logits up to ~ +/- 80)
    extreme = {}
    for dtype_name, dt in [("fp32", torch.float32), ("fp16", torch.float16)]:
        qe = (q * 4.0).to(dt)
        ke = (k * 4.0).to(dt)
        ve = v.to(dt)
        ref_e = dense_causal_attention(qe.float(), ke.float(), ve.float())
        extreme[dtype_name] = {}
        for name, fn in [
            ("dense", dense_causal_attention),
            ("tiled_128", lambda a, b, c: tiled_online_softmax_causal_attention(a, b, c, 128, 128)),
            ("sdpa_default", lambda a, b, c: sdpa_causal_attention(a, b, c, None)),
        ]:
            ye = fn(qe, ke, ve)
            finite = bool(torch.isfinite(ye).all().item())
            diff = (ye.float() - ref_e).abs()
            extreme[dtype_name][name] = {
                "all_finite": finite,
                "max_abs_err_vs_fp32_ref": float(diff.max().item()) if finite else None,
            }

    return {
        "causality_checks": causality,
        "tile_size_checks": tile_checks,
        "extreme_score_checks": extreme,
    }


def probe_sdpa_backends(device: str = "cuda") -> Dict[str, Any]:
    """Probe which SDPA backends are supported on this GPU for FP16 and FP32."""
    B, H, L, d = 2, 4, 512, 64
    backends = [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
        ("MATH", SDPBackend.MATH),
    ]
    report: Dict[str, Any] = {}
    for dtype_name, dt in [("fp16", torch.float16), ("fp32", torch.float32)]:
        q = torch.randn(B, H, L, d, device=device, dtype=dt, requires_grad=True)
        k = torch.randn(B, H, L, d, device=device, dtype=dt, requires_grad=True)
        v = torch.randn(B, H, L, d, device=device, dtype=dt, requires_grad=True)
        report[dtype_name] = {}
        for bname, b_enum in backends:
            try:
                with sdpa_kernel(b_enum):
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    out.sum().backward()
                torch.cuda.synchronize()
                report[dtype_name][bname] = {"supported": True, "error": None}
            except Exception as exc:
                report[dtype_name][bname] = {"supported": False, "error": str(exc)[:220]}
            finally:
                q.grad = k.grad = v.grad = None
    return report


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg


def benchmark_operator_condition(
    fn,
    B: int,
    H: int,
    L: int,
    d: int,
    dtype: torch.dtype,
    ref_pack: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    q_base_fp32: torch.Tensor,
    k_base_fp32: torch.Tensor,
    v_base_fp32: torch.Tensor,
    dout_fp32: torch.Tensor,
    warmup: int = 3,
    reps: int = 10,
    run_backward: bool = True,
) -> Dict[str, Any]:
    """Measure forward-only, forward+backward timing/memory, and output/gradient error vs FP32 reference."""
    res: Dict[str, Any] = {
        "B": B, "H": H, "L": L, "d_h": d, "dtype": str(dtype).replace("torch.", ""),
        "status": "OK",
        "forward": None,
        "forward_backward": None,
        "accuracy_vs_dense_fp32": None,
    }

    q = q_base_fp32.to(dtype=dtype)
    k = k_base_fp32.to(dtype=dtype)
    v = v_base_fp32.to(dtype=dtype)
    dout = dout_fp32.to(dtype=dtype)

    # 1. Forward-only benchmark
    clear_cuda()
    try:
        def _fwd():
            with torch.no_grad():
                return fn(q, k, v)
        res["forward"] = measure_cuda_fn(_fwd, warmup=warmup, reps=reps, reset_peak_per_rep=True)
    except Exception as exc:
        clear_cuda()
        res["status"] = "OOM_FWD" if _is_oom(exc) else f"ERR_FWD: {str(exc)[:180]}"
        return res

    # 2. Output numerical error (when dense FP32 reference is feasible)
    acc: Dict[str, Any] = {}
    if ref_pack is not None:
        ref_out, ref_dq, ref_dk, ref_dv = ref_pack
        with torch.no_grad():
            out_val = fn(q, k, v).float()
        diff_o = (out_val - ref_out).abs()
        acc["out_max_abs_err"] = float(diff_o.max().item())
        acc["out_mean_abs_err"] = float(diff_o.mean().item())
        del out_val, diff_o

    # 3. Forward + Backward benchmark and gradient error
    if run_backward:
        clear_cuda()
        try:
            def _fwdbwd():
                qg = q.detach().requires_grad_(True)
                kg = k.detach().requires_grad_(True)
                vg = v.detach().requires_grad_(True)
                out = fn(qg, kg, vg)
                out.backward(dout)
                return qg.grad, kg.grad, vg.grad

            res["forward_backward"] = measure_cuda_fn(_fwdbwd, warmup=warmup, reps=reps, reset_peak_per_rep=True)
            if ref_pack is not None:
                _, ref_dq, ref_dk, ref_dv = ref_pack
                dq, dk, dv = _fwdbwd()
                acc["dQ_max_abs_err"] = float((dq.float() - ref_dq).abs().max().item())
                acc["dK_max_abs_err"] = float((dk.float() - ref_dk).abs().max().item())
                acc["dV_max_abs_err"] = float((dv.float() - ref_dv).abs().max().item())
                acc["dQ_mean_abs_err"] = float((dq.float() - ref_dq).abs().mean().item())
                acc["dK_mean_abs_err"] = float((dk.float() - ref_dk).abs().mean().item())
                acc["dV_mean_abs_err"] = float((dv.float() - ref_dv).abs().mean().item())
                del dq, dk, dv
        except Exception as exc:
            clear_cuda()
            res["forward_backward"] = {
                "status": "OOM_BWD" if _is_oom(exc) else f"ERR_BWD: {str(exc)[:180]}",
            }
            if res["status"] == "OK":
                res["status"] = "FWD_OK_BWD_OOM" if _is_oom(exc) else "FWD_OK_BWD_ERR"

    if acc:
        res["accuracy_vs_dense_fp32"] = acc
    clear_cuda()
    return res


def run_part_a(
    data_root: Path,
    out_dir: Path,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    set_seed(6304)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    B, H, d = 2, 4, 64

    lengths = [256, 512] if smoke_test else [256, 512, 1024, 2048, 4096, 8192]
    reps = 2 if smoke_test else 10
    warmup = 1 if smoke_test else 3
    dtypes = [("fp16", torch.float16), ("fp32", torch.float32)]

    print(f"[Task 2A] Running causality, tile-size, extreme-score & SDPA backend checks...", flush=True)
    checks = run_causality_and_edge_checks(device=device)
    backend_support = probe_sdpa_backends(device=device)
    print(f"[Task 2A] SDPA backend probe: {backend_support}", flush=True)

    # Determine which SDPA backends are supported per dtype so we benchmark default + forced backends cleanly
    rows: List[Dict[str, Any]] = []

    for L in lengths:
        clear_cuda()
        gen = torch.Generator(device=device).manual_seed(6304 + L)
        q_fp32 = torch.randn(B, H, L, d, generator=gen, device=device, dtype=torch.float32)
        k_fp32 = torch.randn(B, H, L, d, generator=gen, device=device, dtype=torch.float32)
        v_fp32 = torch.randn(B, H, L, d, generator=gen, device=device, dtype=torch.float32)
        dout_fp32 = torch.randn(B, H, L, d, generator=gen, device=device, dtype=torch.float32)

        # Compute dense FP32 reference at feasible lengths (<= 4096 on 16GB T4)
        ref_pack = None
        if L <= 4096:
            try:
                qg = q_fp32.detach().clone().requires_grad_(True)
                kg = k_fp32.detach().clone().requires_grad_(True)
                vg = v_fp32.detach().clone().requires_grad_(True)
                out_ref = dense_causal_attention(qg, kg, vg)
                out_ref.backward(dout_fp32)
                ref_pack = (
                    out_ref.detach().clone(),
                    qg.grad.detach().clone(),
                    kg.grad.detach().clone(),
                    vg.grad.detach().clone(),
                )
                del qg, kg, vg, out_ref
                clear_cuda()
            except Exception:
                ref_pack = None
                clear_cuda()

        for dtype_name, dt in dtypes:
            methods = [
                ("A1_dense", dense_causal_attention, None),
                ("A2_tiled_128", lambda a, b, c: tiled_online_softmax_causal_attention(a, b, c, 128, 128),
                 "python_tiled_online_softmax_autograd_stores_tile_intermediates"),
                ("A3_sdpa_default", lambda a, b, c: sdpa_causal_attention(a, b, c, None), "sdpa_dispatched_default"),
            ]
            # Also benchmark forced EFFICIENT_ATTENTION and MATH when supported on this GPU
            for bname, benum in [("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION), ("MATH", SDPBackend.MATH)]:
                if backend_support[dtype_name][bname]["supported"]:
                    methods.append((
                        f"A3_sdpa_{bname.lower()}",
                        lambda a, b, c, be=benum: sdpa_causal_attention(a, b, c, be),
                        f"forced_{bname}",
                    ))

            for mname, fn, note in methods:
                # Avoid wasting time on Python tiled backward at L=8192 if it is very slow/OOM; still test forward!
                run_bwd = not (mname == "A2_tiled_128" and L >= 8192)
                rec = benchmark_operator_condition(
                    fn, B, H, L, d, dt, ref_pack, q_fp32, k_fp32, v_fp32, dout_fp32,
                    warmup=warmup, reps=reps, run_backward=run_bwd,
                )
                rec["method"] = mname
                rec["note"] = note
                if not run_bwd:
                    rec["forward_backward"] = {"status": "SKIPPED_PYTHON_TILED_BWD_AT_8192"}
                rows.append(rec)
                fwd_ms = rec["forward"]["mean_ms"] if rec["forward"] else float("nan")
                bwd_ms = (
                    rec["forward_backward"]["mean_ms"]
                    if isinstance(rec["forward_backward"], dict) and "mean_ms" in rec["forward_backward"]
                    else float("nan")
                )
                mem_mb = (
                    rec["forward"]["peak_allocated_bytes"] / 1e6 if rec["forward"] else 0.0
                )
                print(
                    f"  [2A] L={L:4d} {dtype_name:4s} {mname:26s} | status={rec['status']:14s} | "
                    f"fwd={fwd_ms:7.2f}ms | fwd+bwd={bwd_ms:8.2f}ms | fwd_peak={mem_mb:6.1f}MB",
                    flush=True,
                )

        del q_fp32, k_fp32, v_fp32, dout_fp32, ref_pack
        clear_cuda()

    payload = {
        "metadata": collect_run_metadata(data_root, extra={"task": "task2_part_a_attention", "smoke_test": smoke_test}),
        "checks": checks,
        "sdpa_backend_support": backend_support,
        "benchmarks": rows,
    }
    save_json(out_dir / "part_a_results.json", payload)
    print(f"[Task 2A] Saved {out_dir / 'part_a_results.json'}", flush=True)
    return payload
