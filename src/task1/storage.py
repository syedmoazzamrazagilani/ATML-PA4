from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

ELIGIBLE_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def get_eligible_modules(model: nn.Module) -> Dict[str, nn.Linear]:
    """Return ordered dict of all eligible Transformer block linear projections."""
    eligible: Dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(name.endswith(s) for s in ELIGIBLE_SUFFIXES):
            eligible[name] = mod
    assert len(eligible) > 0, "No eligible projections found!"
    return eligible


def verify_model_parameter_accounting(model: nn.Module) -> Dict[str, int]:
    """Enumerate unique parameters (counting tied weights once) and verify eligible set."""
    eligible = get_eligible_modules(model)
    eligible_ptrs = {m.weight.data_ptr() for m in eligible.values()}

    seen_ptrs = set()
    total_unique_params = 0
    total_unique_fp16_bytes = 0
    eligible_params = 0
    uncompressed_params = 0
    uncompressed_fp16_bytes = 0
    tied_duplicates = 0

    for name, p in model.named_parameters():
        ptr = p.data_ptr()
        if ptr in seen_ptrs:
            tied_duplicates += 1
            continue
        seen_ptrs.add(ptr)
        n = p.numel()
        b = n * 2  # FP16 reference representation
        total_unique_params += n
        total_unique_fp16_bytes += b
        if ptr in eligible_ptrs:
            eligible_params += n
        else:
            uncompressed_params += n
            uncompressed_fp16_bytes += b

    return {
        "num_eligible_projections": len(eligible),
        "total_unique_params": total_unique_params,
        "total_unique_fp16_bytes": total_unique_fp16_bytes,
        "eligible_params": eligible_params,
        "eligible_fp16_bytes": eligible_params * 2,
        "uncompressed_params": uncompressed_params,
        "uncompressed_fp16_bytes": uncompressed_fp16_bytes,
        "tied_parameter_tensors_deduplicated": tied_duplicates,
    }


def compute_sparse_storage_bytes(
    model: nn.Module,
    acct: Dict[str, int],
) -> Dict[str, float | int]:
    """
    Unstructured 16-bit sparse model per PA4 spec:
      1 mask bit per eligible weight (byte-aligned per matrix) + 16 bits per nonzero,
      plus uncompressed FP16 parameters (tied weights counted once).
    """
    eligible = get_eligible_modules(model)
    total_elig = 0
    total_nnz = 0
    mask_bytes = 0
    val_bytes = 0

    for mod in eligible.values():
        w = mod.weight.data
        n = w.numel()
        nnz = int((w != 0).sum().item())
        total_elig += n
        total_nnz += nnz
        mask_bytes += math.ceil(n / 8)
        val_bytes += nnz * 2

    elig_sparse_bytes = mask_bytes + val_bytes
    total_sparse_bytes = acct["uncompressed_fp16_bytes"] + elig_sparse_bytes
    achieved_sparsity = 1.0 - (total_nnz / total_elig)
    return {
        "eligible_params": total_elig,
        "eligible_nonzeros": total_nnz,
        "eligible_zeros": total_elig - total_nnz,
        "achieved_sparsity": achieved_sparsity,
        "achieved_nonzero_fraction": total_nnz / total_elig,
        "sparse_mask_bytes": mask_bytes,
        "sparse_value_bytes": val_bytes,
        "eligible_sparse_bytes": elig_sparse_bytes,
        "uncompressed_fp16_bytes": acct["uncompressed_fp16_bytes"],
        "total_packed_bytes": total_sparse_bytes,
        "dense_fp16_bytes": acct["total_unique_fp16_bytes"],
        "compression_ratio_vs_fp16": acct["total_unique_fp16_bytes"] / total_sparse_bytes,
        "nominal_bits_per_eligible_weight": (elig_sparse_bytes * 8.0) / total_elig,
    }


def projection_quant_packed_bytes(
    d_out: int,
    d_in: int,
    bits: int,
    group_size: int = 128,
    extra_fp16_channel_scales: int = 0,
) -> Dict[str, int]:
    """
    Exact packed storage for a single group-wise asymmetric quantized projection:
      - Groups along input columns (d_in), padded to multiple of group_size if needed.
      - Bit-packed b-bit integers (byte-aligned per matrix).
      - Per-group metadata: FP16 scale (2 bytes) + uint8 zero-point (1 byte) = 3 bytes/group.
      - Optional serialized compensating FP16 channel scales (e.g., unfolded AWQ o_proj).
    """
    n_groups_per_row = math.ceil(d_in / group_size)
    padded_in = n_groups_per_row * group_size
    pad_elems = d_out * (padded_in - d_in)
    n_groups = d_out * n_groups_per_row
    padded_weights = d_out * padded_in

    payload_bytes = math.ceil((padded_weights * bits) / 8)
    metadata_bytes = n_groups * 3  # 2B FP16 scale + 1B uint8 zero-point
    extra_scale_bytes = extra_fp16_channel_scales * 2

    return {
        "d_out": d_out,
        "d_in": d_in,
        "bits": bits,
        "group_size": group_size,
        "num_groups": n_groups,
        "padding_elements": pad_elems,
        "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes,
        "extra_scale_bytes": extra_scale_bytes,
        "total_bytes": payload_bytes + metadata_bytes + extra_scale_bytes,
    }


def compute_quant_storage_bytes(
    eligible_shapes: Dict[str, Tuple[int, int]],
    bit_map: Dict[str, int],
    acct: Dict[str, int],
    group_size: int = 128,
    unfolded_scale_counts: Dict[str, int] | None = None,
) -> Dict[str, float | int]:
    """Compute exact total packed storage across all eligible projections + uncompressed params."""
    unfolded_scale_counts = unfolded_scale_counts or {}
    payload_bytes = 0
    metadata_bytes = 0
    extra_scale_bytes = 0
    padding_elements = 0
    total_elig = 0
    weighted_bits = 0

    for name, (d_out, d_in) in eligible_shapes.items():
        b = int(bit_map[name])
        info = projection_quant_packed_bytes(
            d_out, d_in, b, group_size=group_size,
            extra_fp16_channel_scales=unfolded_scale_counts.get(name, 0),
        )
        payload_bytes += info["payload_bytes"]
        metadata_bytes += info["metadata_bytes"]
        extra_scale_bytes += info["extra_scale_bytes"]
        padding_elements += info["padding_elements"]
        total_elig += d_out * d_in
        weighted_bits += (d_out * d_in) * b

    elig_packed = payload_bytes + metadata_bytes + extra_scale_bytes
    total_packed = acct["uncompressed_fp16_bytes"] + elig_packed
    return {
        "eligible_params": total_elig,
        "nominal_bits_per_eligible_weight": weighted_bits / total_elig,
        "effective_bits_per_eligible_weight": (elig_packed * 8.0) / total_elig,
        "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes,
        "extra_scale_bytes": extra_scale_bytes,
        "padding_elements": padding_elements,
        "eligible_packed_bytes": elig_packed,
        "uncompressed_fp16_bytes": acct["uncompressed_fp16_bytes"],
        "total_packed_bytes": total_packed,
        "dense_fp16_bytes": acct["total_unique_fp16_bytes"],
        "compression_ratio_vs_fp16": acct["total_unique_fp16_bytes"] / total_packed,
        "quant_convention": "asymmetric_group128_fp16scale_uint8zp_bitpacked",
    }
