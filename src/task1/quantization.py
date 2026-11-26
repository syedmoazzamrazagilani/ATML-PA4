from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.task1.storage import get_eligible_modules, projection_quant_packed_bytes


def quantize_dequantize_groupwise_asym(
    weight: torch.Tensor,
    bits: int,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Asymmetric group-wise weight-only PTQ along input columns (group_size=128):
      scale = (w_max - w_min) / (2^b - 1)   [stored FP16]
      zp    = clip(round(-w_min / scale), 0, 2^b - 1) [stored uint8]
      q     = clip(round(w / scale) + zp, 0, 2^b - 1)
      w_hat = scale * (q - zp)              [reconstructed FP16]
    """
    orig_dtype = weight.dtype
    d_out, d_in = weight.shape
    w = weight.float()
    pad = (group_size - (d_in % group_size)) % group_size
    if pad > 0:
        w = torch.nn.functional.pad(w, (0, pad), value=0.0)

    n_groups = w.shape[1] // group_size
    wg = w.view(d_out, n_groups, group_size)

    qmax = float((1 << bits) - 1)
    w_min = wg.amin(dim=-1, keepdim=True)
    w_max = wg.amax(dim=-1, keepdim=True)

    scale = ((w_max - w_min) / qmax).clamp_min(1e-8).to(torch.float16).float()
    zp = torch.clamp(torch.round(-w_min / scale), 0.0, qmax)

    q = torch.clamp(torch.round(wg / scale) + zp, 0.0, qmax)
    w_hat = (scale * (q - zp)).view(d_out, -1)
    if pad > 0:
        w_hat = w_hat[:, :d_in]
    return w_hat.to(orig_dtype)


@torch.no_grad()
def apply_q1_uniform_ptq(
    model: nn.Module,
    bits: int,
    group_size: int = 128,
) -> Dict[str, int]:
    """Apply Q1 uniform asymmetric group-wise PTQ to all eligible projections."""
    eligible = get_eligible_modules(model)
    bit_map: Dict[str, int] = {}
    for name, mod in eligible.items():
        mod.weight.data.copy_(quantize_dequantize_groupwise_asym(mod.weight.data, bits=bits, group_size=group_size))
        bit_map[name] = bits
    return bit_map


def _awq_scale_from_alpha(x_abs_mean: torch.Tensor, alpha: float) -> torch.Tensor:
    s = x_abs_mean.clamp_min(1e-5).pow(alpha)
    s = s / torch.sqrt(s.max() * s.min()).clamp_min(1e-5)
    return s


@torch.no_grad()
def search_awq_group_scale(
    weights: List[torch.Tensor],
    bits_list: List[int],
    x_sample: torch.Tensor,
    x_abs_mean: torch.Tensor,
    group_size: int = 128,
    alpha_grid: List[float] | None = None,
) -> Tuple[float, torch.Tensor, float]:
    """
    Search AWQ exponent alpha in [0, 1] minimizing total output MSE across one or more
    projections sharing the same input activations.
    """
    if alpha_grid is None:
        alpha_grid = [round(i * 0.05, 2) for i in range(21)]

    x = x_sample.to(device=weights[0].device, dtype=torch.float16)
    x_mean = x_abs_mean.to(device=weights[0].device, dtype=torch.float32)

    refs = [x @ w.t() for w in weights]
    best_alpha = 0.0
    best_s = torch.ones_like(x_mean)
    best_loss = float("inf")

    for alpha in alpha_grid:
        s = _awq_scale_from_alpha(x_mean, alpha)
        inv_s_fp16 = (1.0 / s).to(torch.float16)
        x_scaled = x * inv_s_fp16.unsqueeze(0)

        loss = 0.0
        for w, b, y_ref in zip(weights, bits_list, refs):
            w_scaled = (w.float() * s.unsqueeze(0)).to(torch.float16)
            w_q = quantize_dequantize_groupwise_asym(w_scaled, bits=b, group_size=group_size)
            y_hat = x_scaled @ w_q.t()
            loss += float((y_ref - y_hat).float().pow(2).mean().item())

        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
            best_s = s.detach().clone()

    return best_alpha, best_s, best_loss


@torch.no_grad()
def apply_awq_with_bitmap(
    model: nn.Module,
    bit_map: Dict[str, int],
    x_abs_mean: Dict[str, torch.Tensor],
    x_samples: Dict[str, torch.Tensor],
    group_size: int = 128,
    alpha_grid: List[float] | None = None,
    verify_pre_quant_calib_ids: torch.Tensor | None = None,
) -> Dict[str, Any]:
    """
    Apply AWQ-style per-channel activation scaling + inverse compensation + Q1 quantizer.
    Algebraic folding:
      - q_proj, k_proj, v_proj: shared scale folded into input_layernorm (exact, 0 extra bytes)
      - down_proj: scale folded into up_proj rows BEFORE quantizing up_proj (exact, 0 extra bytes)
      - gate_proj, up_proj: shared scale folded into post_attention_layernorm (exact, 0 extra bytes)
      - o_proj: GQA (12 Q / 2 KV heads) prevents exact unconstrained folding into v_proj;
        compensating FP16 inverse scales are serialized and counted in matched byte budget.
    """
    if alpha_grid is None:
        alpha_grid = [round(i * 0.05, 2) for i in range(21)]

    ref_logits = None
    if verify_pre_quant_calib_ids is not None:
        ref_logits = model(input_ids=verify_pre_quant_calib_ids, use_cache=False).logits.float().cpu()

    chosen_alphas: Dict[str, float] = {}
    unfolded_scale_counts: Dict[str, int] = {}
    o_proj_scales: Dict[str, torch.Tensor] = {}

    # Phase 1: Search and apply exact full-precision rescaling/folding (NO quantization yet)
    for layer_idx, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{layer_idx}"
        q_name = f"{prefix}.self_attn.q_proj"
        k_name = f"{prefix}.self_attn.k_proj"
        v_name = f"{prefix}.self_attn.v_proj"
        o_name = f"{prefix}.self_attn.o_proj"
        g_name = f"{prefix}.mlp.gate_proj"
        u_name = f"{prefix}.mlp.up_proj"
        d_name = f"{prefix}.mlp.down_proj"

        # 1a. q/k/v shared scale -> fold inverse into input_layernorm
        a_qkv, s_qkv, _ = search_awq_group_scale(
            [layer.self_attn.q_proj.weight.data, layer.self_attn.k_proj.weight.data, layer.self_attn.v_proj.weight.data],
            [bit_map[q_name], bit_map[k_name], bit_map[v_name]],
            x_samples[q_name], x_abs_mean[q_name], group_size=group_size, alpha_grid=alpha_grid,
        )
        chosen_alphas[f"{prefix}.qkv_shared"] = a_qkv
        layer.input_layernorm.weight.data.copy_(
            (layer.input_layernorm.weight.data.float() / s_qkv).to(torch.float16)
        )
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            proj.weight.data.copy_((proj.weight.data.float() * s_qkv.unsqueeze(0)).to(torch.float16))

        # 1b. o_proj scale -> serialized compensating FP16 inverse scale
        a_o, s_o, _ = search_awq_group_scale(
            [layer.self_attn.o_proj.weight.data], [bit_map[o_name]],
            x_samples[o_name], x_abs_mean[o_name], group_size=group_size, alpha_grid=alpha_grid,
        )
        chosen_alphas[o_name] = a_o
        o_proj_scales[o_name] = s_o
        unfolded_scale_counts[o_name] = int(layer.self_attn.o_proj.in_features)

        # 1c. down_proj scale -> fold inverse into up_proj rows BEFORE gate/up folding
        a_d, s_d, _ = search_awq_group_scale(
            [layer.mlp.down_proj.weight.data], [bit_map[d_name]],
            x_samples[d_name], x_abs_mean[d_name], group_size=group_size, alpha_grid=alpha_grid,
        )
        chosen_alphas[d_name] = a_d
        layer.mlp.up_proj.weight.data.copy_(
            (layer.mlp.up_proj.weight.data.float() / s_d.unsqueeze(1)).to(torch.float16)
        )
        if layer.mlp.up_proj.bias is not None:
            layer.mlp.up_proj.bias.data.copy_(
                (layer.mlp.up_proj.bias.data.float() / s_d).to(torch.float16)
            )
        layer.mlp.down_proj.weight.data.copy_(
            (layer.mlp.down_proj.weight.data.float() * s_d.unsqueeze(0)).to(torch.float16)
        )

        # 1d. gate/up shared scale -> fold inverse into post_attention_layernorm
        a_gu, s_gu, _ = search_awq_group_scale(
            [layer.mlp.gate_proj.weight.data, layer.mlp.up_proj.weight.data],
            [bit_map[g_name], bit_map[u_name]],
            x_samples[g_name], x_abs_mean[g_name], group_size=group_size, alpha_grid=alpha_grid,
        )
        chosen_alphas[f"{prefix}.gate_up_shared"] = a_gu
        layer.post_attention_layernorm.weight.data.copy_(
            (layer.post_attention_layernorm.weight.data.float() / s_gu).to(torch.float16)
        )
        layer.mlp.gate_proj.weight.data.copy_(
            (layer.mlp.gate_proj.weight.data.float() * s_gu.unsqueeze(0)).to(torch.float16)
        )
        layer.mlp.up_proj.weight.data.copy_(
            (layer.mlp.up_proj.weight.data.float() * s_gu.unsqueeze(0)).to(torch.float16)
        )

    # Verify full-precision folding equivalence BEFORE quantization rounding
    equiv_stats = None
    if verify_pre_quant_calib_ids is not None and ref_logits is not None:
        folded_logits = model(input_ids=verify_pre_quant_calib_ids, use_cache=False).logits.float().cpu()
        diff = (ref_logits - folded_logits).abs()
        equiv_stats = {
            "pre_quant_max_abs_logit_diff": float(diff.max().item()),
            "pre_quant_mean_abs_logit_diff": float(diff.mean().item()),
        }

    # Phase 2: Quantize all eligible projections with the identical Q1 quantizer
    eligible = get_eligible_modules(model)
    for name, mod in eligible.items():
        b = int(bit_map[name])
        if name in o_proj_scales:
            s_o = o_proj_scales[name]
            w_scaled = (mod.weight.data.float() * s_o.unsqueeze(0)).to(torch.float16)
            w_q = quantize_dequantize_groupwise_asym(w_scaled, bits=b, group_size=group_size)
            mod.weight.data.copy_((w_q.float() / s_o.unsqueeze(0)).to(torch.float16))
        else:
            mod.weight.data.copy_(quantize_dequantize_groupwise_asym(mod.weight.data, bits=b, group_size=group_size))

    return {
        "alpha_grid": alpha_grid,
        "chosen_alphas": chosen_alphas,
        "unfolded_scale_counts": unfolded_scale_counts,
        "pre_quant_equivalence": equiv_stats,
    }


def compute_diagonal_fisher_calibration(
    model: nn.Module,
    calib_rows: List[Dict[str, Any]],
    num_seqs: int = 64,
    loss_scale: float = 512.0,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Compute diagonal empirical Fisher information E[(dL/dW)^2] on calibration sequences
    for HAWQ/Molchanov-inspired mixed-precision quantization sensitivity.
    Uses fixed loss scaling to prevent FP16 gradient underflow and guards against non-finite grads.
    """
    model.eval()
    eligible = get_eligible_modules(model)
    for p in model.parameters():
        p.requires_grad_(False)
    for mod in eligible.values():
        if mod.weight.is_inference():
            mod.weight = nn.Parameter(mod.weight.detach().clone(), requires_grad=True)
        else:
            mod.weight.requires_grad_(True)

    fisher = {n: torch.zeros_like(mod.weight, dtype=torch.float32, device="cpu") for n, mod in eligible.items()}
    used = min(num_seqs, len(calib_rows))
    valid_steps = 0

    with torch.enable_grad():
        for i in range(used):
            ids = torch.tensor([calib_rows[i]["token_ids"]], dtype=torch.long, device=device)
            model.zero_grad(set_to_none=True)
            out = model(input_ids=ids, labels=ids, use_cache=False)
            (out.loss * loss_scale).backward()

            step_ok = True
            for mod in eligible.values():
                if mod.weight.grad is None or not torch.isfinite(mod.weight.grad).all():
                    step_ok = False
                    break
            if not step_ok:
                continue

            valid_steps += 1
            for name, mod in eligible.items():
                g = mod.weight.grad.detach().float() / loss_scale
                fisher[name] += g.pow(2).cpu()

    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(False)

    denom = float(max(valid_steps, 1))
    for name in fisher:
        fisher[name] /= denom
    return fisher


@torch.no_grad()
def compute_quant_sensitivity_tables(
    model: nn.Module,
    orig_weights: Dict[str, torch.Tensor],
    x_abs_mean: Dict[str, torch.Tensor],
    x_samples: Dict[str, torch.Tensor],
    fisher_diag: Dict[str, torch.Tensor],
    allowed_bits: Tuple[int, ...] = (2, 3, 4, 8),
    group_size: int = 128,
    device: str = "cuda",
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    For every eligible projection and every allowed bit width b in {2, 3, 4, 8},
    compute two AWQ-quantization error sensitivity tables on calibration data:
      1. awq_output_mse: local AWQ calibration output MSE ||W X - W_hat X||_F^2 / N
      2. awq_fisher_loss: 0.5 * sum(F_ij * (W_ij - W_hat_ij)^2) (second-order loss impact)
    """
    eligible = get_eligible_modules(model)
    mse_table: Dict[str, Dict[int, float]] = {n: {} for n in eligible}
    fisher_table: Dict[str, Dict[int, float]] = {n: {} for n in eligible}
    coarse_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for name, mod in eligible.items():
        w = orig_weights[name + ".weight"].to(device=device, dtype=torch.float16)
        x = x_samples[name].to(device=device, dtype=torch.float16)
        x_m = x_abs_mean[name].to(device=device, dtype=torch.float32)
        f_diag = fisher_diag[name].to(device=device, dtype=torch.float32)

        for b in allowed_bits:
            _, s, mse_val = search_awq_group_scale(
                [w], [b], x, x_m, group_size=group_size, alpha_grid=coarse_grid,
            )
            w_scaled = (w.float() * s.unsqueeze(0)).to(torch.float16)
            w_q = quantize_dequantize_groupwise_asym(w_scaled, bits=b, group_size=group_size)
            w_eff = w_q.float() / s.unsqueeze(0)
            delta_sq = (w.float() - w_eff).pow(2)
            fisher_err = float(0.5 * (f_diag * delta_sq).sum().item())

            mse_table[name][b] = float(mse_val)
            fisher_table[name][b] = fisher_err

    return {
        "awq_output_mse": mse_table,
        "awq_fisher_loss": fisher_table,
    }


def solve_mixed_precision_knapsack(
    eligible_shapes: Dict[str, Tuple[int, int]],
    error_table: Dict[str, Dict[int, float]],
    target_eligible_bytes: int,
    unfolded_scale_counts: Dict[str, int],
    allowed_bits: Tuple[int, ...] = (2, 3, 4, 8),
    group_size: int = 128,
) -> Dict[str, int]:
    """
    Greedy marginal-error-reduction-per-byte Multiple-Choice Knapsack solver.
    Starts all projections at min(allowed_bits) and upgrades projections by highest
    (error_reduction / extra_packed_bytes) strictly obeying sum(packed_bytes) <= target_eligible_bytes.
    """
    bits_sorted = sorted(allowed_bits)
    b_min = bits_sorted[0]

    costs: Dict[str, Dict[int, int]] = {}
    for name, (d_out, d_in) in eligible_shapes.items():
        extra = unfolded_scale_counts.get(name, 0)
        costs[name] = {
            b: projection_quant_packed_bytes(
                d_out, d_in, b, group_size=group_size, extra_fp16_channel_scales=extra
            )["total_bytes"]
            for b in bits_sorted
        }

    bit_map = {name: b_min for name in eligible_shapes}
    used_bytes = sum(costs[n][b_min] for n in eligible_shapes)
    assert used_bytes <= target_eligible_bytes, (
        f"Even min bit width {b_min} ({used_bytes} B) exceeds target ({target_eligible_bytes} B)"
    )

    while True:
        best_gain_per_byte = -1.0
        best_move: Tuple[str, int, int] | None = None
        rem = target_eligible_bytes - used_bytes

        for name in eligible_shapes:
            cur_b = bit_map[name]
            cur_c = costs[name][cur_b]
            cur_e = error_table[name][cur_b]
            for nxt_b in bits_sorted:
                if nxt_b <= cur_b:
                    continue
                dc = costs[name][nxt_b] - cur_c
                if dc <= 0 or dc > rem:
                    continue
                de = cur_e - error_table[name][nxt_b]
                if de <= 0:
                    continue
                ratio = de / float(dc)
                if ratio > best_gain_per_byte:
                    best_gain_per_byte = ratio
                    best_move = (name, nxt_b, dc)

        if best_move is None:
            break
        m_name, m_bit, m_dc = best_move
        bit_map[m_name] = m_bit
        used_bytes += m_dc

    assert used_bytes <= target_eligible_bytes
    return bit_map
