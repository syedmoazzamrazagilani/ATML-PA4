from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.task1.storage import get_eligible_modules


@torch.inference_mode()
def collect_calibration_stats(
    model: nn.Module,
    calib_rows: List[Dict[str, Any]],
    sample_tokens_per_seq: int = 16,
    batch_size: int = 4,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Single pass over the 128 x 512 C4 calibration sequences collecting:
      - x_l2_norm[name]: L2 norm of each input channel across all 65,536 tokens (for Wanda)
      - x_abs_mean[name]: Mean |x| per input channel across all 65,536 tokens (for AWQ)
      - x_samples[name]: Compact CPU FP16 activation sample [2048, d_in] (for AWQ search & error probes)
    """
    model.eval()
    eligible = get_eligible_modules(model)
    x_sq_sum: Dict[str, torch.Tensor] = {}
    x_abs_sum: Dict[str, torch.Tensor] = {}
    x_samples: Dict[str, List[torch.Tensor]] = {n: [] for n in eligible}
    total_tokens = 0

    # Deterministic stride of 16 tokens per 512-token sequence (2,048 tokens total across 128 seqs)
    sample_idx = torch.linspace(0, 511, steps=sample_tokens_per_seq).long()

    handles = []
    for name, mod in eligible.items():
        d_in = mod.in_features
        x_sq_sum[name] = torch.zeros(d_in, dtype=torch.float64, device=device)
        x_abs_sum[name] = torch.zeros(d_in, dtype=torch.float64, device=device)

        def _make_hook(proj_name: str):
            def _hook(_mod, inputs):
                x = inputs[0]  # [B, T, d_in]
                flat = x.reshape(-1, x.shape[-1]).float()
                x_sq_sum[proj_name] += flat.pow(2).sum(dim=0).double()
                x_abs_sum[proj_name] += flat.abs().sum(dim=0).double()
                sub = x[:, sample_idx.to(x.device), :].reshape(-1, x.shape[-1]).to(torch.float16).cpu()
                x_samples[proj_name].append(sub)
            return _hook

        handles.append(mod.register_forward_pre_hook(_make_hook(name)))

    for start in range(0, len(calib_rows), batch_size):
        batch = calib_rows[start : start + batch_size]
        ids = torch.tensor([r["token_ids"] for r in batch], dtype=torch.long, device=device)
        total_tokens += ids.numel()
        _ = model(input_ids=ids, use_cache=False)

    for h in handles:
        h.remove()

    x_l2_norm = {n: torch.sqrt(x_sq_sum[n].clamp_min(1e-12)).float().cpu() for n in eligible}
    x_abs_mean = {n: (x_abs_sum[n] / float(total_tokens)).float().cpu() for n in eligible}
    x_cat = {n: torch.cat(x_samples[n], dim=0) for n in eligible}
    return {
        "total_tokens": total_tokens,
        "x_l2_norm": x_l2_norm,
        "x_abs_mean": x_abs_mean,
        "x_samples": x_cat,
    }


def snapshot_model_weights_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot eligible projection weights and layer norms on CPU for fast lossless resets."""
    snap: Dict[str, torch.Tensor] = {}
    eligible = get_eligible_modules(model)
    for name, mod in eligible.items():
        snap[name + ".weight"] = mod.weight.data.detach().clone().cpu()
    for name, mod in model.named_modules():
        if "layernorm" in name.lower() and hasattr(mod, "weight") and mod.weight is not None:
            snap[name + ".weight"] = mod.weight.data.detach().clone().cpu()
    return snap


def restore_model_weights(model: nn.Module, snapshot: Dict[str, torch.Tensor]) -> None:
    """Restore model weights from CPU snapshot in-place."""
    mod_dict = dict(model.named_modules())
    for full_key, tensor_cpu in snapshot.items():
        mod_name, attr = full_key.rsplit(".", 1)
        param = getattr(mod_dict[mod_name], attr)
        param.data.copy_(tensor_cpu.to(device=param.device, dtype=param.dtype))


def _prune_matrix_per_row(
    weight: torch.Tensor,
    score: torch.Tensor,
    zeros_per_row: int | torch.Tensor,
) -> torch.Tensor:
    """Apply unstructured per-output-row pruning given a score matrix."""
    d_out, d_in = weight.shape
    if isinstance(zeros_per_row, int):
        if zeros_per_row <= 0:
            return weight.clone()
        idx = torch.topk(score, k=zeros_per_row, dim=1, largest=False).indices
        out = weight.clone()
        out.scatter_(1, idx, 0)
        return out

    # Row-specific integer zero counts (used only when matching exact global budget)
    out = weight.clone()
    order = torch.argsort(score, dim=1, descending=False)
    for r in range(d_out):
        k = int(zeros_per_row[r].item())
        if k > 0:
            out[r, order[r, :k]] = 0
    return out


def compute_calibration_layer_errors(
    orig_weights: Dict[str, torch.Tensor],
    model: nn.Module,
    x_samples: Dict[str, torch.Tensor],
    device: str = "cuda",
) -> Dict[str, Any]:
    """Compute per-projection relative L2 output error ||WX - W_hat X||_F / ||WX||_F on calibration samples."""
    eligible = get_eligible_modules(model)
    per_proj: Dict[str, float] = {}
    sq_err_sum = 0.0
    sq_ref_sum = 0.0

    for name, mod in eligible.items():
        x = x_samples[name].to(device=device, dtype=torch.float16)  # [N, d_in]
        w_orig = orig_weights[name + ".weight"].to(device=device, dtype=torch.float16)
        w_cur = mod.weight.data
        with torch.inference_mode():
            y_ref = x @ w_orig.t()
            y_diff = x @ (w_orig - w_cur).t()
            err_sq = float(y_diff.float().pow(2).sum().item())
            ref_sq = float(y_ref.float().pow(2).sum().item())
        rel_l2 = math.sqrt(err_sq / max(ref_sq, 1e-12))
        per_proj[name] = rel_l2
        sq_err_sum += err_sq
        sq_ref_sum += ref_sq

    vals = list(per_proj.values())
    return {
        "mean_rel_l2": float(np.mean(vals)),
        "median_rel_l2": float(np.median(vals)),
        "max_rel_l2": float(np.max(vals)),
        "pooled_rel_l2": math.sqrt(sq_err_sum / max(sq_ref_sum, 1e-12)),
        "per_projection_rel_l2": per_proj,
    }


def apply_uniform_pruning(
    model: nn.Module,
    sparsity: float,
    method: str,
    x_l2_norm: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, int]:
    """Apply P1 (magnitude) or P2 (Wanda) uniform per-output-row unstructured pruning."""
    assert method in ("magnitude", "wanda")
    eligible = get_eligible_modules(model)
    total_zeros = 0
    total_weights = 0

    for name, mod in eligible.items():
        w = mod.weight.data
        d_out, d_in = w.shape
        k = int(round(sparsity * d_in))
        if method == "magnitude":
            score = w.float().abs()
        else:
            assert x_l2_norm is not None
            norm = x_l2_norm[name].to(device=w.device, dtype=torch.float32)
            score = w.float().abs() * norm.unsqueeze(0)
        mod.weight.data = _prune_matrix_per_row(w, score, k)
        total_zeros += d_out * k
        total_weights += d_out * d_in

    return {"target_zeros": total_zeros, "total_eligible": total_weights}


def apply_wanda_2_4(
    model: nn.Module,
    x_l2_norm: Dict[str, torch.Tensor],
) -> Dict[str, int]:
    """Apply structured 2:4 Wanda pruning (retain top-2 in every group of 4 contiguous input channels)."""
    eligible = get_eligible_modules(model)
    total_zeros = 0
    total_weights = 0

    for name, mod in eligible.items():
        w = mod.weight.data
        d_out, d_in = w.shape
        assert d_in % 4 == 0, f"{name} d_in={d_in} not divisible by 4"
        norm = x_l2_norm[name].to(device=w.device, dtype=torch.float32)
        score = (w.float().abs() * norm.unsqueeze(0)).view(d_out, d_in // 4, 4)
        idx = torch.topk(score, k=2, dim=-1, largest=False).indices
        w_grouped = w.clone().view(d_out, d_in // 4, 4)
        w_grouped.scatter_(-1, idx, 0)
        mod.weight.data = w_grouped.view(d_out, d_in)
        total_zeros += (d_out * d_in) // 2
        total_weights += d_out * d_in

    return {"target_zeros": total_zeros, "total_eligible": total_weights}


def compute_pruning_sensitivities(
    model: nn.Module,
    orig_weights: Dict[str, torch.Tensor],
    x_l2_norm: Dict[str, torch.Tensor],
    x_samples: Dict[str, torch.Tensor],
    target_sparsity: float,
    owl_m: float = 5.0,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """
    Compute two literature-grounded sensitivity probes per eligible projection:
      1. owl_outlier_ratio: Yin et al. (2024) fraction of Wanda scores > M * mean(score).
      2. calib_distortion: local Wanda pruning output distortion ||WX - W_pruned X||_F / ||WX||_F.
    Higher value = more sensitive = should receive LOWER sparsity.
    """
    eligible = get_eligible_modules(model)
    owl_scores: Dict[str, float] = {}
    dist_scores: Dict[str, float] = {}

    for name, mod in eligible.items():
        w = orig_weights[name + ".weight"].to(device=device, dtype=torch.float16)
        norm = x_l2_norm[name].to(device=device, dtype=torch.float32)
        score = w.float().abs() * norm.unsqueeze(0)

        # 1. OWL outlier ratio
        thresh = owl_m * score.mean()
        owl_scores[name] = float((score > thresh).float().mean().item())

        # 2. Local calibration output distortion at target sparsity
        k = int(round(target_sparsity * w.shape[1]))
        w_pruned = _prune_matrix_per_row(w, score, k)
        x = x_samples[name].to(device=device, dtype=torch.float16)
        with torch.inference_mode():
            y_ref = x @ w.t()
            y_diff = x @ (w - w_pruned).t()
            rel_err = math.sqrt(
                float(y_diff.float().pow(2).sum().item())
                / max(float(y_ref.float().pow(2).sum().item()), 1e-12)
            )
        dist_scores[name] = float(rel_err)

    return {
        "owl_outlier_ratio": owl_scores,
        "calib_distortion": dist_scores,
    }


def allocate_nonuniform_sparsities(
    eligible_shapes: Dict[str, Tuple[int, int]],
    sensitivity: Dict[str, float],
    target_sparsity: float,
    lam: float,
    use_log_sensitivity: bool = True,
    s_min: float = 0.10,
    s_max: float = 0.85,
) -> Tuple[Dict[str, float], Dict[str, int], int]:
    """
    Convert per-projection sensitivity values into nonuniform sparsities in [0.10, 0.85]
    matching the EXACT total zeroed eligible weight count of uniform pruning.

    Rule: s_l = clip(target_sparsity - lam * z_l + delta, s_min, s_max),
    where z_l is standardized sensitivity (higher sensitivity -> lower sparsity) and delta is
    found by bisection so sum_l n_l * s_l = target_zeros.
    """
    names = list(eligible_shapes.keys())
    counts = np.array([eligible_shapes[n][0] * eligible_shapes[n][1] for n in names], dtype=np.float64)
    target_zeros = int(sum(eligible_shapes[n][0] * int(round(target_sparsity * eligible_shapes[n][1])) for n in names))

    raw = np.array([sensitivity[n] for n in names], dtype=np.float64)
    if use_log_sensitivity:
        raw = np.log(np.clip(raw, 1e-12, None))
    std = float(raw.std())
    z = (raw - raw.mean()) / (std if std > 1e-12 else 1.0)
    # Clip extreme z-scores softly to prevent degenerate saturation
    z = np.clip(z, -2.5, 2.5)

    # Bisection on delta to match exact continuous zero count
    lo, hi = -1.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s_vec = np.clip(target_sparsity - lam * z + mid, s_min, s_max)
        if float((s_vec * counts).sum()) < target_zeros:
            lo = mid
        else:
            hi = mid
    s_cont = np.clip(target_sparsity - lam * z + 0.5 * (lo + hi), s_min, s_max)

    # Convert to per-row integer zero counts (bounded in [ceil(0.10*d_in), floor(0.85*d_in)])
    row_zeros: Dict[str, int] = {}
    min_k: Dict[str, int] = {}
    max_k: Dict[str, int] = {}
    for idx, n in enumerate(names):
        d_out, d_in = eligible_shapes[n]
        lo_k = math.ceil(s_min * d_in)
        hi_k = math.floor(s_max * d_in)
        min_k[n] = lo_k
        max_k[n] = hi_k
        row_zeros[n] = int(np.clip(round(float(s_cont[idx]) * d_in), lo_k, hi_k))

    # Fine-adjust per-row counts to get as close as possible to target_zeros
    cur_zeros = sum(eligible_shapes[n][0] * row_zeros[n] for n in names)
    # Sort smallest d_out first for fine granularity
    sorted_by_rows = sorted(names, key=lambda n: eligible_shapes[n][0])
    for _ in range(500):
        diff = target_zeros - cur_zeros
        if abs(diff) < eligible_shapes[sorted_by_rows[0]][0]:
            break
        improved = False
        for n in sorted_by_rows:
            d_out = eligible_shapes[n][0]
            if diff > 0 and row_zeros[n] < max_k[n] and d_out <= diff:
                row_zeros[n] += 1
                cur_zeros += d_out
                diff -= d_out
                improved = True
            elif diff < 0 and row_zeros[n] > min_k[n] and d_out <= -diff:
                row_zeros[n] -= 1
                cur_zeros -= d_out
                diff += d_out
                improved = True
        if not improved:
            break

    residual = target_zeros - cur_zeros
    achieved_map = {n: row_zeros[n] / eligible_shapes[n][1] for n in names}
    return achieved_map, row_zeros, residual


def apply_nonuniform_wanda(
    model: nn.Module,
    x_l2_norm: Dict[str, torch.Tensor],
    row_zeros: Dict[str, int],
    residual_single_weights: int = 0,
) -> Dict[str, Any]:
    """
    Apply nonuniform Wanda pruning using per-projection row_zeros.
    If residual_single_weights != 0 (smaller than smallest d_out), adjusts single rows
    of a flexible projection so exact global zero count matches P1/P2 to the single weight.
    """
    eligible = get_eligible_modules(model)
    rem = int(residual_single_weights)
    actual_sparsities: Dict[str, float] = {}
    total_zeros = 0
    total_weights = 0

    for name, mod in eligible.items():
        w = mod.weight.data
        d_out, d_in = w.shape
        norm = x_l2_norm[name].to(device=w.device, dtype=torch.float32)
        score = w.float().abs() * norm.unsqueeze(0)
        k_base = int(row_zeros[name])

        if rem != 0:
            lo_k = math.ceil(0.10 * d_in)
            hi_k = math.floor(0.85 * d_in)
            if rem > 0 and k_base < hi_k and rem <= d_out:
                k_vec = torch.full((d_out,), k_base, dtype=torch.long)
                k_vec[:rem] += 1
                mod.weight.data = _prune_matrix_per_row(w, score, k_vec)
                nz = int(k_vec.sum().item())
                rem = 0
            elif rem < 0 and k_base > lo_k and (-rem) <= d_out:
                k_vec = torch.full((d_out,), k_base, dtype=torch.long)
                k_vec[:-rem] -= 1
                mod.weight.data = _prune_matrix_per_row(w, score, k_vec)
                nz = int(k_vec.sum().item())
                rem = 0
            else:
                mod.weight.data = _prune_matrix_per_row(w, score, k_base)
                nz = d_out * k_base
        else:
            mod.weight.data = _prune_matrix_per_row(w, score, k_base)
            nz = d_out * k_base

        actual_sparsities[name] = nz / (d_out * d_in)
        total_zeros += nz
        total_weights += d_out * d_in

    assert rem == 0, f"Unresolved residual zeros: {rem}"
    return {
        "total_zeros": total_zeros,
        "total_eligible": total_weights,
        "achieved_global_sparsity": total_zeros / total_weights,
        "per_projection_sparsity": actual_sparsities,
    }
