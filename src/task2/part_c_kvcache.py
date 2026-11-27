from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from src.common.data_loader import PA4Data, read_jsonl
from src.common.env import (
    collect_run_metadata,
    get_kv_layer,
    num_cache_layers,
    save_json,
    set_kv_layer,
    set_seed,
)
from src.common.metrics import paired_bootstrap_mean_diff
from src.common.timing import clear_cuda, measure_cuda_fn


GROUP_SIZE = 64
SINK_TOKENS = 8
RECENT_TOKENS = 64
MAX_NEW_TOKENS = 32


# =============================================================================
# Cache & RoPE utilities (transformers==4.57.1 compatible)
# =============================================================================

def clone_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    new_cache = DynamicCache()
    for i in range(num_cache_layers(cache)):
        k, v = get_kv_layer(cache, i)
        new_cache.update(k.clone(), v.clone(), i)
    return new_cache


def crop_dynamic_cache(cache: DynamicCache, max_len: int) -> None:
    for i in range(num_cache_layers(cache)):
        k, v = get_kv_layer(cache, i)
        set_kv_layer(cache, i, k[:, :, :max_len, :].contiguous(), v[:, :, :max_len, :].contiguous())


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope_q(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # q: [B, H_q, L_q, d_h]; cos, sin: [B, L_q, d_h]
    return (q * cos.unsqueeze(1)) + (_rotate_half(q) * sin.unsqueeze(1))


# =============================================================================
# Exact Serialized Byte Accounting (Matches PA4 Task 2C Formulas)
# =============================================================================

def fp16_kv_bytes(L: int, n_layers: int, n_kv_heads: int, head_dim: int) -> int:
    return 4 * n_layers * n_kv_heads * head_dim * L


def quant_kv_bytes(L: int, bits: int, n_layers: int, n_kv_heads: int, head_dim: int, g: int = GROUP_SIZE) -> int:
    """
    M_b(L) = 2 * n_l * H_kv * d_h * [ floor(L/g) * (g*b/8 + 4) + 2*(L mod g) ]
    Metadata per 64-element group: FP16 scale (2B) + FP16 offset (2B) = 4 bytes.
    """
    channels = 2 * n_layers * n_kv_heads * head_dim
    n_full_groups = L // g
    rem = L % g
    bytes_per_group = (g * bits) // 8 + 4
    return channels * (n_full_groups * bytes_per_group + 2 * rem)


def max_fp16_positions_for_budget(
    budget_bytes: int,
    L_ctx: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    index_bytes_per_pos: int = 4,
) -> Tuple[int, int]:
    """
    Largest legal number of globally shared FP16 positions whose actual serialized bytes
    (FP16 KV payload + int32 position indices) do not exceed budget_bytes.
    """
    bytes_per_tok = fp16_kv_bytes(1, n_layers, n_kv_heads, head_dim) + index_bytes_per_pos
    k_keep = min(L_ctx, budget_bytes // bytes_per_tok)
    k_keep = max(k_keep, min(L_ctx, SINK_TOKENS + RECENT_TOKENS))
    actual_bytes = k_keep * bytes_per_tok
    assert actual_bytes <= budget_bytes, f"Eviction bytes {actual_bytes} > budget {budget_bytes}"
    return int(k_keep), int(actual_bytes)


def solve_hybrid_allocation(
    budget_bytes: int,
    L_ctx: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    g: int = GROUP_SIZE,
    index_bytes_per_pos: int = 4,
) -> Dict[str, int]:
    """
    C4 Hybrid:
      - n_int2 = g * m positions stored in channel-K/token-V INT2 (complete 64-token groups).
      - n_fp16 = L_ctx - n_int2 positions stored in FP16 (>= 8 sink + 64 recent, absorbing residual).
      - Includes int32 position indices for all L_ctx positions (4 * L_ctx bytes).
    Choose smallest integer m >= 0 (maximizing n_fp16) with total_bytes <= budget_bytes.
    """
    channels = 2 * n_layers * n_kv_heads * head_dim
    int2_group_bytes = (g * 2) // 8 + 4  # 16 + 4 = 20 bytes per 64-element group
    idx_bytes = L_ctx * index_bytes_per_pos
    max_m = max(0, (L_ctx - (SINK_TOKENS + RECENT_TOKENS)) // g)

    chosen = None
    for m in range(0, max_m + 1):
        n_int2 = m * g
        n_fp16 = L_ctx - n_int2
        total_b = channels * (m * int2_group_bytes + 2 * n_fp16) + idx_bytes
        if total_b <= budget_bytes:
            chosen = {
                "num_int2_groups": m,
                "num_int2_positions": n_int2,
                "num_fp16_positions": n_fp16,
                "index_bytes": idx_bytes,
                "total_bytes": int(total_b),
            }
            break

    assert chosen is not None, f"No feasible C4 hybrid allocation for L={L_ctx}, budget={budget_bytes}"
    return chosen


# =============================================================================
# Prompt Splitting, Evidence Token Mapping, and Deterministic Answer Extractor
# =============================================================================

def prepare_retrieval_case(case: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    """
    Apply Qwen chat template, map gold evidence character span -> token indices (eval only),
    and split prompt into (pre_q_ids, q_suffix_ids) WITHOUT touching evidence coordinates.
    """
    user_text = case["user_text"]
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    user_start_in_full = full_text.find(user_text)
    assert user_start_in_full >= 0, f"Could not locate user_text inside chat template for {case['id']}"

    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids: List[int] = enc["input_ids"]
    offsets: List[Tuple[int, int]] = enc["offset_mapping"]

    # Map evidence char span into token indices (STRICTLY for post-hoc diagnostics only)
    ev_start = user_start_in_full + int(case["evidence_char_start"])
    ev_end = user_start_in_full + int(case["evidence_char_end"])
    evidence_tok_indices = [
        i for i, (s, e) in enumerate(offsets) if e > ev_start and s < ev_end and e > s
    ]

    # Split pre-question context from final question suffix using ONLY structural text markers
    # (never touches evidence_char_start / evidence_char_end).
    split_char_in_user = None
    for sep in ("\n\n", "\n"):
        pos = user_text.rfind(sep)
        if pos != -1:
            tail = user_text[pos + len(sep) :].strip()
            if 10 <= len(tail) <= 450:
                split_char_in_user = pos + len(sep)
                break

    if split_char_in_user is not None:
        split_char_in_full = user_start_in_full + split_char_in_user
        split_tok_idx = next(
            (i for i, (s, e) in enumerate(offsets) if s >= split_char_in_full),
            len(full_ids) - 32,
        )
    else:
        # Fallback: last 32 tokens of user turn + assistant prompt header
        user_end_in_full = user_start_in_full + len(user_text)
        user_end_tok = next(
            (i for i, (s, e) in enumerate(offsets) if s >= user_end_in_full),
            len(full_ids) - 4,
        )
        split_tok_idx = max(128, user_end_tok - 32)

    # Guarantee valid split boundaries
    split_tok_idx = max(SINK_TOKENS + RECENT_TOKENS + 1, min(split_tok_idx, len(full_ids) - 2))

    # Evidence indices inside pre-question context
    ev_in_ctx = [i for i in evidence_tok_indices if i < split_tok_idx]

    ans_str = str(case["code"]).strip()
    ans_ids = tokenizer.encode(ans_str, add_special_tokens=False)

    return {
        "id": case["id"],
        "code": ans_str,
        "target_tokens": int(case["target_tokens"]),
        "position": str(case["position"]),
        "distractors": int(case["distractors"]),
        "full_ids": full_ids,
        "pre_q_ids": full_ids[:split_tok_idx],
        "q_suffix_ids": full_ids[split_tok_idx:],
        "ans_ids": ans_ids,
        "L_full": len(full_ids),
        "L_ctx": split_tok_idx,
        "L_q": len(full_ids) - split_tok_idx,
        "evidence_tok_indices_in_ctx": ev_in_ctx,
        "evidence_tok_count_total": len(evidence_tok_indices),
    }


def extract_five_digit_code(text: str) -> Optional[str]:
    """Fixed deterministic 5-digit code extractor."""
    m = re.findall(r"\b\d{5}\b", text)
    if m:
        return m[0]
    m2 = re.findall(r"\d{5}", text)
    return m2[0] if m2 else None


# =============================================================================
# Question-Aware Importance Scoring (SDPA Kept Active; FP32 Masked Scores)
# =============================================================================

@torch.no_grad()
def compute_question_aware_scores(
    model: torch.nn.Module,
    pre_q_cache: DynamicCache,
    q_suffix_ids: torch.Tensor,
    L_ctx: int,
) -> Tuple[torch.Tensor, float]:
    """
    Compute global SnapKV-inspired importance scores [L_ctx] for pre-question positions
    using query projections from the question/observation suffix against rotary-positioned
    cached keys in FP32, while keeping the model strictly on its working SDPA backend.
    """
    device = q_suffix_ids.device
    L_q = q_suffix_ids.shape[1]
    n_layers = len(model.model.layers)
    n_q_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_values_heads if hasattr(model.config, "num_key_values_heads") else model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_q_heads
    gqa_rep = n_q_heads // n_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    # Pre-read existing rotary-positioned context keys before running question suffix
    ctx_keys = [get_kv_layer(pre_q_cache, i)[0].detach() for i in range(n_layers)]
    importance_acc = torch.zeros(L_ctx, device=device, dtype=torch.float32)

    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        def _make_pre_hook(li: int, attn_mod: torch.nn.Module):
            def _hook(_mod, args, kwargs):
                hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                pos_emb = kwargs.get("position_embeddings", None)
                if pos_emb is None:
                    pos_ids = kwargs.get(
                        "position_ids",
                        torch.arange(L_ctx, L_ctx + L_q, device=device).unsqueeze(0),
                    )
                    pos_emb = model.model.rotary_emb(hidden, pos_ids)
                cos, sin = pos_emb

                B, Tq, _ = hidden.shape
                q_proj = attn_mod.q_proj(hidden).view(B, Tq, n_q_heads, head_dim).transpose(1, 2)
                q_rot = _apply_rope_q(q_proj, cos, sin).float()  # [B, H_q, L_q, d_h]

                k_ctx = ctx_keys[li].float()  # [B, H_kv, L_ctx, d_h]
                if gqa_rep > 1:
                    k_ctx = k_ctx.repeat_interleave(gqa_rep, dim=1)  # [B, H_q, L_ctx, d_h]

                scores = torch.matmul(q_rot, k_ctx.transpose(-2, -1)) * scale  # [B, H_q, L_q, L_ctx]
                probs = F.softmax(scores, dim=-1)
                # Average importance over batch, query heads, and question-suffix tokens
                importance_acc.add_(probs.mean(dim=(0, 1, 2)))
            return _hook

        handles.append(layer.self_attn.register_forward_pre_hook(_make_pre_hook(layer_idx, layer.self_attn), with_kwargs=True))

    # Run question suffix on a temporary clone so pre_q_cache is untouched
    tmp_cache = clone_dynamic_cache(pre_q_cache)
    pos_ids = torch.arange(L_ctx, L_ctx + L_q, device=device).unsqueeze(0)
    cache_pos = torch.arange(L_ctx, L_ctx + L_q, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model(
        input_ids=q_suffix_ids,
        past_key_values=tmp_cache,
        position_ids=pos_ids,
        cache_position=cache_pos,
        use_cache=True,
    )
    torch.cuda.synchronize()
    scoring_ms = (time.perf_counter() - t0) * 1000.0

    for h in handles:
        h.remove()
    del tmp_cache

    importance = importance_acc / float(n_layers)
    assert torch.isfinite(importance).all(), "Non-finite question-aware importance scores!"
    return importance, scoring_ms


# =============================================================================
# Asymmetric Group-64 KV Quantization (FP16 Scale + FP16 Offset)
# =============================================================================

def quant_dequant_tokenwise(x: torch.Tensor, bits: int, g: int = GROUP_SIZE) -> torch.Tensor:
    """
    Token-wise asymmetric quantization along feature dimension (d_h=128 -> 2 groups of 64).
    x: [B, H_kv, L_q, d_h]
    """
    B, H, L, D = x.shape
    assert D % g == 0
    qmax = float((1 << bits) - 1)
    xg = x.float().view(B, H, L, D // g, g)
    x_min = xg.amin(dim=-1, keepdim=True).to(torch.float16).float()  # FP16 offset
    x_max = xg.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / qmax).clamp_min(1e-8).to(torch.float16).float()  # FP16 scale
    q = torch.clamp(torch.round((xg - x_min) / scale), 0.0, qmax)
    x_hat = (q * scale + x_min).view(B, H, L, D)
    return x_hat.to(x.dtype)


def quant_dequant_channelwise(x: torch.Tensor, bits: int, g: int = GROUP_SIZE) -> torch.Tensor:
    """
    Channel-wise asymmetric quantization along token dimension (groups of 64 consecutive tokens per channel).
    x: [B, H_kv, L_q, d_h] with L_q % g == 0.
    """
    B, H, L, D = x.shape
    assert L % g == 0
    qmax = float((1 << bits) - 1)
    xg = x.float().view(B, H, L // g, g, D)
    x_min = xg.amin(dim=-2, keepdim=True).to(torch.float16).float()  # FP16 offset
    x_max = xg.amax(dim=-2, keepdim=True)
    scale = ((x_max - x_min) / qmax).clamp_min(1e-8).to(torch.float16).float()  # FP16 scale
    q = torch.clamp(torch.round((xg - x_min) / scale), 0.0, qmax)
    x_hat = (q * scale + x_min).view(B, H, L, D)
    return x_hat.to(x.dtype)


def quantize_kv_chunk_inplace(
    cache: DynamicCache,
    start_pos: int,
    end_pos: int,
    layout: str,
    bits: int,
    g: int = GROUP_SIZE,
) -> Dict[str, float]:
    """
    Quantize/dequantize cache[:, :, start_pos:end_pos, :] in-place (must satisfy (end_pos - start_pos) % g == 0).
    Returns separate K and V reconstruction MSE and relative L2 errors.
    """
    assert layout in ("tokenK_tokenV", "chanK_tokenV")
    n_tok = end_pos - start_pos
    assert n_tok >= 0 and (n_tok % g == 0)
    if n_tok == 0:
        return {"k_mse": 0.0, "v_mse": 0.0, "k_rel_l2": 0.0, "v_rel_l2": 0.0}

    k_err_sq = k_ref_sq = v_err_sq = v_ref_sq = 0.0
    n_elems = 0
    for i in range(num_cache_layers(cache)):
        k, v = get_kv_layer(cache, i)
        k_sub = k[:, :, start_pos:end_pos, :]
        v_sub = v[:, :, start_pos:end_pos, :]

        k_hat = (
            quant_dequant_tokenwise(k_sub, bits=bits, g=g)
            if layout == "tokenK_tokenV"
            else quant_dequant_channelwise(k_sub, bits=bits, g=g)
        )
        v_hat = quant_dequant_tokenwise(v_sub, bits=bits, g=g)

        k_err_sq += float((k_sub.float() - k_hat.float()).pow(2).sum().item())
        k_ref_sq += float(k_sub.float().pow(2).sum().item())
        v_err_sq += float((v_sub.float() - v_hat.float()).pow(2).sum().item())
        v_ref_sq += float(v_sub.float().pow(2).sum().item())
        n_elems += int(k_sub.numel())

        k[:, :, start_pos:end_pos, :] = k_hat
        v[:, :, start_pos:end_pos, :] = v_hat

    return {
        "k_mse": k_err_sq / max(n_elems, 1),
        "v_mse": v_err_sq / max(n_elems, 1),
        "k_rel_l2": math.sqrt(k_err_sq / max(k_ref_sq, 1e-12)),
        "v_rel_l2": math.sqrt(v_err_sq / max(v_ref_sq, 1e-12)),
    }


def apply_eviction_inplace(cache: DynamicCache, keep_indices: torch.Tensor) -> None:
    """Keep only globally selected positions keep_indices (sorted 1D LongTensor) across all layers."""
    for i in range(num_cache_layers(cache)):
        k, v = get_kv_layer(cache, i)
        set_kv_layer(
            cache, i,
            k.index_select(dim=2, index=keep_indices).contiguous(),
            v.index_select(dim=2, index=keep_indices).contiguous(),
        )


def apply_hybrid_fp16_int2_inplace(
    cache: DynamicCache,
    low_prec_indices: torch.Tensor,
    g: int = GROUP_SIZE,
) -> Dict[str, float]:
    """
    C4 Hybrid: keep FP16 positions untouched; gather the low_prec_indices (len = 64*m, sorted),
    apply channel-K / token-V INT2 quantization to those complete 64-token groups,
    and write dequantized tensors back to their exact original positions.
    """
    n_low = int(low_prec_indices.numel())
    assert n_low % g == 0
    if n_low == 0:
        return {"k_mse": 0.0, "v_mse": 0.0, "k_rel_l2": 0.0, "v_rel_l2": 0.0}

    k_err_sq = k_ref_sq = v_err_sq = v_ref_sq = 0.0
    n_elems = 0
    for i in range(num_cache_layers(cache)):
        k, v = get_kv_layer(cache, i)
        k_low = k.index_select(dim=2, index=low_prec_indices)
        v_low = v.index_select(dim=2, index=low_prec_indices)

        k_hat = quant_dequant_channelwise(k_low, bits=2, g=g)
        v_hat = quant_dequant_tokenwise(v_low, bits=2, g=g)

        k_err_sq += float((k_low.float() - k_hat.float()).pow(2).sum().item())
        k_ref_sq += float(k_low.float().pow(2).sum().item())
        v_err_sq += float((v_low.float() - v_hat.float()).pow(2).sum().item())
        v_ref_sq += float(v_low.float().pow(2).sum().item())
        n_elems += int(k_low.numel())

        k[:, :, low_prec_indices, :] = k_hat
        v[:, :, low_prec_indices, :] = v_hat

    return {
        "k_mse": k_err_sq / max(n_elems, 1),
        "v_mse": v_err_sq / max(n_elems, 1),
        "k_rel_l2": math.sqrt(k_err_sq / max(k_ref_sq, 1e-12)),
        "v_rel_l2": math.sqrt(v_err_sq / max(v_ref_sq, 1e-12)),
    }


# =============================================================================
# Question-Suffix Forward, Reference-Answer NLL, and Greedy Decode
# =============================================================================

@torch.no_grad()
def run_policy_from_ctx_cache(
    model: torch.nn.Module,
    tokenizer: Any,
    cache: DynamicCache,
    q_suffix_ids: torch.Tensor,
    ans_ids: List[int],
    L_ctx_orig: int,
    ref_post_q_logprobs: Optional[torch.Tensor] = None,
    decode_quant_layout: Optional[str] = None,
    decode_quant_bits: int = 4,
    quantized_prefix_len: int = 0,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """
    Advance cache through question suffix (preserving original absolute rotary position IDs),
    compute post-prefill logit KL vs FP16 and teacher-forced reference-answer NLL,
    then run greedy generation with optional incremental 64-token KV quantization.
    """
    device = q_suffix_ids.device
    L_q = q_suffix_ids.shape[1]
    eos_id = tokenizer.eos_token_id
    phys_before_q = cache.get_seq_length()

    # 1. Question-suffix forward (preserving original absolute rotary positions L_ctx_orig .. L_ctx_orig + L_q - 1)
    pos_ids = torch.arange(L_ctx_orig, L_ctx_orig + L_q, device=device).unsqueeze(0)
    cache_pos = torch.arange(phys_before_q, phys_before_q + L_q, device=device)

    torch.cuda.synchronize()
    t_q0 = time.perf_counter()
    out_q = model(
        input_ids=q_suffix_ids,
        past_key_values=cache,
        position_ids=pos_ids,
        cache_position=cache_pos,
        use_cache=True,
    )
    torch.cuda.synchronize()
    q_suffix_ms = (time.perf_counter() - t_q0) * 1000.0

    # If decode quantization is active, quantize any newly completed 64-token blocks right after q_suffix
    q_ptr = quantized_prefix_len
    if decode_quant_layout is not None:
        cur_len = cache.get_seq_length()
        n_new_full = ((cur_len - q_ptr) // GROUP_SIZE) * GROUP_SIZE
        if n_new_full > 0:
            quantize_kv_chunk_inplace(cache, q_ptr, q_ptr + n_new_full, decode_quant_layout, decode_quant_bits)
            q_ptr += n_new_full

    post_q_logits = out_q.logits[:, -1, :].float()  # [1, V]
    assert torch.isfinite(post_q_logits).all(), "Non-finite post-prefill logits!"
    post_q_logprobs = F.log_softmax(post_q_logits, dim=-1)

    kl_from_fp16 = 0.0
    if ref_post_q_logprobs is not None:
        # D_KL(p_fp16 || p_policy)
        p_ref = ref_post_q_logprobs.exp()
        kl_from_fp16 = float((p_ref * (ref_post_q_logprobs - post_q_logprobs)).sum(dim=-1).clamp_min(0.0).item())

    # 2. Teacher-forced reference-answer NLL (exact on gold 5-digit code tokens, then crop cache back)
    phys_post_q = cache.get_seq_length()
    orig_post_q = L_ctx_orig + L_q
    ans_nll_sum = float(-post_q_logprobs[0, ans_ids[0]].item())
    if len(ans_ids) > 1:
        ans_prefix = torch.tensor([ans_ids[:-1]], dtype=torch.long, device=device)
        m_len = ans_prefix.shape[1]
        a_pos_ids = torch.arange(orig_post_q, orig_post_q + m_len, device=device).unsqueeze(0)
        a_cache_pos = torch.arange(phys_post_q, phys_post_q + m_len, device=device)
        out_ans = model(
            input_ids=ans_prefix,
            past_key_values=cache,
            position_ids=a_pos_ids,
            cache_position=a_cache_pos,
            use_cache=True,
        )
        ans_lp = F.log_softmax(out_ans.logits.float(), dim=-1)
        for j in range(m_len):
            ans_nll_sum += float(-ans_lp[0, j, ans_ids[j + 1]].item())
        crop_dynamic_cache(cache, phys_post_q)
    ref_answer_nll = ans_nll_sum / float(len(ans_ids))

    # 3. Free-running greedy decode
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_d0 = time.perf_counter()

    generated_ids: List[int] = []
    next_tok_id = int(post_q_logits.argmax(dim=-1).item())
    cur_orig = orig_post_q
    cur_phys = phys_post_q

    for _ in range(max_new_tokens):
        generated_ids.append(next_tok_id)
        if next_tok_id == eos_id:
            break
        if len(generated_ids) >= max_new_tokens:
            break

        inp = torch.tensor([[next_tok_id]], dtype=torch.long, device=device)
        p_ids = torch.tensor([[cur_orig]], dtype=torch.long, device=device)
        c_pos = torch.tensor([cur_phys], dtype=torch.long, device=device)
        out_step = model(
            input_ids=inp,
            past_key_values=cache,
            position_ids=p_ids,
            cache_position=c_pos,
            use_cache=True,
        )
        cur_orig += 1
        cur_phys += 1

        if decode_quant_layout is not None and (cur_phys - q_ptr) >= GROUP_SIZE:
            quantize_kv_chunk_inplace(cache, q_ptr, q_ptr + GROUP_SIZE, decode_quant_layout, decode_quant_bits)
            q_ptr += GROUP_SIZE

        next_tok_id = int(out_step.logits[:, -1, :].argmax(dim=-1).item())

    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - t_d0) * 1000.0
    peak_alloc = int(torch.cuda.max_memory_allocated())

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "generated_ids": generated_ids,
        "generated_text": gen_text,
        "extracted_code": extract_five_digit_code(gen_text),
        "num_generated_tokens": len(generated_ids),
        "ref_answer_nll": ref_answer_nll,
        "post_prefill_logit_kl": kl_from_fp16,
        "q_suffix_ms": q_suffix_ms,
        "decode_ms": decode_ms,
        "peak_allocated_bytes": peak_alloc,
        "post_q_logprobs": post_q_logprobs,  # popped before JSON save
    }


# =============================================================================
# C1 Validation: Full-Forward vs Split-Prefill & 100%-Retained Cache Equivalence
# =============================================================================

@torch.no_grad()
def validate_c1_reference_cache(
    model: torch.nn.Module,
    tokenizer: Any,
    prep_case: Dict[str, Any],
    device: str = "cuda",
) -> Dict[str, Any]:
    full_ids = torch.tensor([prep_case["full_ids"]], dtype=torch.long, device=device)
    pre_q_ids = torch.tensor([prep_case["pre_q_ids"]], dtype=torch.long, device=device)
    q_suffix_ids = torch.tensor([prep_case["q_suffix_ids"]], dtype=torch.long, device=device)
    L_ctx = prep_case["L_ctx"]
    L_q = prep_case["L_q"]

    # 1. Direct full-forward logits
    logits_full = model(input_ids=full_ids, use_cache=False).logits[:, -1, :].float()

    # 2. Split prefill + question-suffix logits
    out_ctx = model(input_ids=pre_q_ids, use_cache=True)
    cache_split = out_ctx.past_key_values
    out_q = model(
        input_ids=q_suffix_ids,
        past_key_values=cache_split,
        position_ids=torch.arange(L_ctx, L_ctx + L_q, device=device).unsqueeze(0),
        cache_position=torch.arange(L_ctx, L_ctx + L_q, device=device),
        use_cache=True,
    )
    logits_split = out_q.logits[:, -1, :].float()
    diff = (logits_full - logits_split).abs()
    split_max_diff = float(diff.max().item())
    split_mean_diff = float(diff.mean().item())
    top1_agree = bool(int(logits_full.argmax(-1).item()) == int(logits_split.argmax(-1).item()))

    # 3. 100%-retained cache greedy generation vs model.generate()
    gen_hf = model.generate(
        full_ids,
        max_new_tokens=16,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )[0, full_ids.shape[1] :].tolist()

    cache_100 = clone_dynamic_cache(out_ctx.past_key_values)
    # Note: out_ctx.past_key_values was mutated by out_q; re-prefill cleanly
    cache_100 = model(input_ids=pre_q_ids, use_cache=True).past_key_values
    res_100 = run_policy_from_ctx_cache(
        model, tokenizer, cache_100, q_suffix_ids, prep_case["ans_ids"], L_ctx, max_new_tokens=16,
    )
    gen_exact_match = (res_100["generated_ids"] == gen_hf)

    # 4. Evicted-cache rotary-position sanity check (keep 8 sink + 64 recent -> verify finite & valid)
    cache_ev = model(input_ids=pre_q_ids, use_cache=True).past_key_values
    keep_idx = torch.cat([
        torch.arange(0, SINK_TOKENS, device=device),
        torch.arange(L_ctx - RECENT_TOKENS, L_ctx, device=device),
    ])
    apply_eviction_inplace(cache_ev, keep_idx)
    res_ev = run_policy_from_ctx_cache(
        model, tokenizer, cache_ev, q_suffix_ids, prep_case["ans_ids"], L_ctx, max_new_tokens=8,
    )

    check = {
        "case_id": prep_case["id"],
        "L_full": prep_case["L_full"],
        "L_ctx": L_ctx,
        "L_q": L_q,
        "split_vs_full_max_abs_logit_diff": split_max_diff,
        "split_vs_full_mean_abs_logit_diff": split_mean_diff,
        "split_vs_full_top1_agree": top1_agree,
        "retained_100pct_vs_hf_generate_exact_match": gen_exact_match,
        "hf_generate_tokens": gen_hf,
        "retained_100pct_tokens": res_100["generated_ids"],
        "evicted_rotary_sanity_ok": len(res_ev["generated_ids"]) > 0 and math.isfinite(res_ev["ref_answer_nll"]),
    }
    assert top1_agree, f"C1 split-prefill top-1 mismatch! max_diff={split_max_diff}"
    assert gen_exact_match, f"C1 100%-cache generation mismatch: {res_100['generated_ids']} vs {gen_hf}"
    print(
        f"[Task 2C C1 Check] PASS | split max_diff={split_max_diff:.5f} | "
        f"100% cache == HF generate: {gen_exact_match} | evicted rotary OK: {check['evicted_rotary_sanity_ok']}",
        flush=True,
    )
    return check


# =============================================================================
# C5 Independent Packed-Cache Encode/Decode Microbenchmark (B in {1, 8}, L in {2048, 4096})
# =============================================================================

def pack_int4_chanK_tokenV(k: torch.Tensor, v: torch.Tensor, g: int = GROUP_SIZE):
    """Actual bit-packed uint8 encoding (2 INT4 values per byte) + FP16 scales & offsets."""
    # k, v: [n_l, B, H_kv, L, d_h]
    n_l, B, H, L, D = k.shape
    L_q = (L // g) * g
    k_main, k_rem = k[..., :L_q, :], k[..., L_q:, :].contiguous()
    v_main, v_rem = v[..., :L_q, :], v[..., L_q:, :].contiguous()

    # Channel-wise K: group along token axis -> [n_l, B, H, L_q//g, g, D]
    kg = k_main.float().view(n_l, B, H, L_q // g, g, D)
    k_min = kg.amin(dim=-2, keepdim=True).to(torch.float16)
    k_max = kg.amax(dim=-2, keepdim=True)
    k_scale = ((k_max - k_min.float()) / 15.0).clamp_min(1e-8).to(torch.float16)
    kq = torch.clamp(torch.round((kg - k_min.float()) / k_scale.float()), 0, 15).to(torch.uint8)
    # Pack 2 adjacent 4-bit values along the g axis -> g//2 bytes
    kq_packed = (kq[..., 0::2, :] | (kq[..., 1::2, :] << 4)).contiguous()

    # Token-wise V: group along feature axis -> [n_l, B, H, L_q, D//g, g]
    vg = v_main.float().view(n_l, B, H, L_q, D // g, g)
    v_min = vg.amin(dim=-1, keepdim=True).to(torch.float16)
    v_max = vg.amax(dim=-1, keepdim=True)
    v_scale = ((v_max - v_min.float()) / 15.0).clamp_min(1e-8).to(torch.float16)
    vq = torch.clamp(torch.round((vg - v_min.float()) / v_scale.float()), 0, 15).to(torch.uint8)
    vq_packed = (vq[..., 0::2] | (vq[..., 1::2] << 4)).contiguous()

    return (kq_packed, k_scale, k_min, k_rem, vq_packed, v_scale, v_min, v_rem, L)


def unpack_int4_chanK_tokenV(packed, g: int = GROUP_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    kq_packed, k_scale, k_min, k_rem, vq_packed, v_scale, v_min, v_rem, L = packed
    n_l, B, H, n_gk, g_half, D = kq_packed.shape
    L_q = n_gk * g

    k_lo = (kq_packed & 0x0F).to(torch.float16)
    k_hi = ((kq_packed >> 4) & 0x0F).to(torch.float16)
    k_unp = torch.stack((k_lo, k_hi), dim=-2).view(n_l, B, H, n_gk, g, D)
    k_main = (k_unp * k_scale + k_min).view(n_l, B, H, L_q, D)
    k_full = torch.cat([k_main, k_rem], dim=-2) if k_rem.shape[-2] > 0 else k_main

    v_lo = (vq_packed & 0x0F).to(torch.float16)
    v_hi = ((vq_packed >> 4) & 0x0F).to(torch.float16)
    v_unp = torch.stack((v_lo, v_hi), dim=-1).view(n_l, B, H, L_q, D // g, g)
    v_main = (v_unp * v_scale + v_min).view(n_l, B, H, L_q, D)
    v_full = torch.cat([v_main, v_rem], dim=-2) if v_rem.shape[-2] > 0 else v_main
    return k_full, v_full


def pack_int2_chanK_tokenV(k: torch.Tensor, v: torch.Tensor, g: int = GROUP_SIZE):
    """Actual bit-packed uint8 encoding (4 INT2 values per byte) + FP16 scales & offsets."""
    n_l, B, H, L, D = k.shape
    L_q = (L // g) * g
    k_main, k_rem = k[..., :L_q, :], k[..., L_q:, :].contiguous()
    v_main, v_rem = v[..., :L_q, :], v[..., L_q:, :].contiguous()

    kg = k_main.float().view(n_l, B, H, L_q // g, g, D)
    k_min = kg.amin(dim=-2, keepdim=True).to(torch.float16)
    k_max = kg.amax(dim=-2, keepdim=True)
    k_scale = ((k_max - k_min.float()) / 3.0).clamp_min(1e-8).to(torch.float16)
    kq = torch.clamp(torch.round((kg - k_min.float()) / k_scale.float()), 0, 3).to(torch.uint8)
    kq_packed = (
        kq[..., 0::4, :] | (kq[..., 1::4, :] << 2) | (kq[..., 2::4, :] << 4) | (kq[..., 3::4, :] << 6)
    ).contiguous()

    vg = v_main.float().view(n_l, B, H, L_q, D // g, g)
    v_min = vg.amin(dim=-1, keepdim=True).to(torch.float16)
    v_max = vg.amax(dim=-1, keepdim=True)
    v_scale = ((v_max - v_min.float()) / 3.0).clamp_min(1e-8).to(torch.float16)
    vq = torch.clamp(torch.round((vg - v_min.float()) / v_scale.float()), 0, 3).to(torch.uint8)
    vq_packed = (
        vq[..., 0::4] | (vq[..., 1::4] << 2) | (vq[..., 2::4] << 4) | (vq[..., 3::4] << 6)
    ).contiguous()
    return (kq_packed, k_scale, k_min, k_rem, vq_packed, v_scale, v_min, v_rem, L)


def unpack_int2_chanK_tokenV(packed, g: int = GROUP_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    kq_packed, k_scale, k_min, k_rem, vq_packed, v_scale, v_min, v_rem, L = packed
    n_l, B, H, n_gk, g_qtr, D = kq_packed.shape
    L_q = n_gk * g

    k0 = (kq_packed & 0x03).to(torch.float16)
    k1 = ((kq_packed >> 2) & 0x03).to(torch.float16)
    k2 = ((kq_packed >> 4) & 0x03).to(torch.float16)
    k3 = ((kq_packed >> 6) & 0x03).to(torch.float16)
    k_unp = torch.stack((k0, k1, k2, k3), dim=-2).view(n_l, B, H, n_gk, g, D)
    k_main = (k_unp * k_scale + k_min).view(n_l, B, H, L_q, D)
    k_full = torch.cat([k_main, k_rem], dim=-2) if k_rem.shape[-2] > 0 else k_main

    v0 = (vq_packed & 0x03).to(torch.float16)
    v1 = ((vq_packed >> 2) & 0x03).to(torch.float16)
    v2 = ((vq_packed >> 4) & 0x03).to(torch.float16)
    v3 = ((vq_packed >> 6) & 0x03).to(torch.float16)
    v_unp = torch.stack((v0, v1, v2, v3), dim=-1).view(n_l, B, H, L_q, D // g, g)
    v_main = (v_unp * v_scale + v_min).view(n_l, B, H, L_q, D)
    v_full = torch.cat([v_main, v_rem], dim=-2) if v_rem.shape[-2] > 0 else v_main
    return k_full, v_full


def _packed_tensor_bytes(packed_tuple) -> int:
    return sum(int(t.numel() * t.element_size()) for t in packed_tuple if isinstance(t, torch.Tensor))


@torch.no_grad()
def run_packed_cache_microbenchmark(
    n_layers: int = 28,
    n_q_heads: int = 12,
    n_kv_heads: int = 2,
    head_dim: int = 128,
    smoke_test: bool = False,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    C5 Independent microbenchmark at B in {1, 8} and L in {2048, 4096}:
    Measures actual bit-packing time, unpacking/dequantization time, serialized tensor bytes,
    and full 28-layer single-token decode attention step latency (with unpacking vs FP16 reference).
    """
    batches = [1] if smoke_test else [1, 8]
    lengths = [2048] if smoke_test else [2048, 4096]
    reps = 3 if smoke_test else 10
    warmup = 1 if smoke_test else 3
    gqa_rep = n_q_heads // n_kv_heads

    rows = []
    for B in batches:
        for L in lengths:
            clear_cuda()
            gen = torch.Generator(device=device).manual_seed(6304 + B * 10000 + L)
            k = torch.randn((n_layers, B, n_kv_heads, L, head_dim), generator=gen, device=device, dtype=torch.float16)
            v = torch.randn((n_layers, B, n_kv_heads, L, head_dim), generator=gen, device=device, dtype=torch.float16)
            q_step = torch.randn((n_layers, B, n_q_heads, 1, head_dim), generator=gen, device=device, dtype=torch.float16)

            fp16_bytes = int(k.numel() * 2 + v.numel() * 2)

            # FP16 reference single-token decode attention across all 28 layers
            def _decode_fp16():
                acc = 0.0
                for li in range(n_layers):
                    kl = k[li].repeat_interleave(gqa_rep, dim=1)
                    vl = v[li].repeat_interleave(gqa_rep, dim=1)
                    out = F.scaled_dot_product_attention(q_step[li], kl, vl, is_causal=False)
                    acc += float(out[0, 0, 0, 0].item())
                return acc

            # Avoid host sync inside loop: return stacked outputs instead
            def _decode_fp16_gpu():
                outs = []
                for li in range(n_layers):
                    kl = k[li].repeat_interleave(gqa_rep, dim=1)
                    vl = v[li].repeat_interleave(gqa_rep, dim=1)
                    outs.append(F.scaled_dot_product_attention(q_step[li], kl, vl, is_causal=False))
                return outs[-1]

            fp16_decode_stats = measure_cuda_fn(_decode_fp16_gpu, warmup=warmup, reps=reps, reset_peak_per_rep=True)

            for scheme, pack_fn, unpack_fn, bits in [
                ("INT4_chanK_tokenV_g64", pack_int4_chanK_tokenV, unpack_int4_chanK_tokenV, 4),
                ("INT2_chanK_tokenV_g64", pack_int2_chanK_tokenV, unpack_int2_chanK_tokenV, 2),
            ]:
                clear_cuda()
                pack_stats = measure_cuda_fn(lambda: pack_fn(k, v), warmup=warmup, reps=reps, reset_peak_per_rep=True)
                packed = pack_fn(k, v)
                ser_bytes = _packed_tensor_bytes(packed)
                theo_bytes = B * quant_kv_bytes(L, bits, n_layers, n_kv_heads, head_dim)
                assert ser_bytes == theo_bytes, f"Serialized bytes mismatch: {ser_bytes} vs {theo_bytes}"

                unpack_stats = measure_cuda_fn(lambda: unpack_fn(packed), warmup=warmup, reps=reps, reset_peak_per_rep=True)

                def _decode_with_unpack():
                    ku, vu = unpack_fn(packed)
                    outs = []
                    for li in range(n_layers):
                        kl = ku[li].repeat_interleave(gqa_rep, dim=1)
                        vl = vu[li].repeat_interleave(gqa_rep, dim=1)
                        outs.append(F.scaled_dot_product_attention(q_step[li], kl, vl, is_causal=False))
                    return outs[-1]

                dec_unp_stats = measure_cuda_fn(_decode_with_unpack, warmup=warmup, reps=reps, reset_peak_per_rep=True)

                rec = {
                    "B": B,
                    "L": L,
                    "scheme": scheme,
                    "bits": bits,
                    "fp16_raw_bytes": fp16_bytes,
                    "serialized_packed_bytes": ser_bytes,
                    "compression_ratio": fp16_bytes / float(ser_bytes),
                    "pack_ms": pack_stats["mean_ms"],
                    "pack_std_ms": pack_stats["std_ms"],
                    "unpack_ms": unpack_stats["mean_ms"],
                    "unpack_std_ms": unpack_stats["std_ms"],
                    "fp16_decode_step_ms": fp16_decode_stats["mean_ms"],
                    "packed_decode_step_with_unpack_ms": dec_unp_stats["mean_ms"],
                    "decode_step_slowdown_vs_fp16": dec_unp_stats["mean_ms"] / max(fp16_decode_stats["mean_ms"], 1e-6),
                    "fp16_decode_peak_alloc_bytes": fp16_decode_stats["peak_allocated_bytes"],
                    "packed_decode_peak_alloc_bytes": dec_unp_stats["peak_allocated_bytes"],
                    "reps": reps,
                }
                rows.append(rec)
                print(
                    f"  [2C Microbench] B={B} L={L} {scheme:22s} | ser={ser_bytes/1e6:5.1f}MB ({rec['compression_ratio']:.2f}x) | "
                    f"pack={rec['pack_ms']:5.2f}ms unpack={rec['unpack_ms']:5.2f}ms | "
                    f"dec_fp16={rec['fp16_decode_step_ms']:5.2f}ms vs dec+unp={rec['packed_decode_step_with_unpack_ms']:5.2f}ms",
                    flush=True,
                )
                del packed
            del k, v, q_step
            clear_cuda()

    return rows


# =============================================================================
# Main Task 2C Runner
# =============================================================================

POLICY_NAMES = [
    "C1_full_fp16",
    "C2_sink_recent_fp16",
    "C2_question_aware_fp16",
    "C3_int4_tokenK_tokenV",
    "C3_int4_chanK_tokenV",
    "C4_hybrid_fp16_int2",
]


def run_part_c(
    data_root: Path,
    out_dir: Path,
    smoke_test: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    set_seed(6304)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    data = PA4Data(data_root)
    model_id, model_rev = data.model_spec("model_target")
    print(f"[Task 2C] Loading {model_id} @ {model_rev} in FP16 (attn=sdpa)...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_rev, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_rev,
        dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device).eval()

    n_layers = model.config.num_hidden_layers
    n_q_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_q_heads

    raw_cases = data.kv_retrieval_cases()
    if smoke_test:
        # Pick 6 diverse cases covering both lengths and all 3 positions
        chosen = []
        seen = set()
        for c in raw_cases:
            key = (c["target_tokens"], c["position"])
            if key not in seen:
                seen.add(key)
                chosen.append(c)
        raw_cases = chosen

    prepared_cases = [prepare_retrieval_case(c, tokenizer) for c in raw_cases]
    print(
        f"[Task 2C] Prepared {len(prepared_cases)} cases | "
        f"L_full range=[{min(p['L_full'] for p in prepared_cases)}, {max(p['L_full'] for p in prepared_cases)}]",
        flush=True,
    )

    # Step C1 Validation on the first case
    c1_validation = validate_c1_reference_cache(model, tokenizer, prepared_cases[0], device=device)

    # Resume support via per-prompt JSONL log
    log_path = out_dir / "part_c_per_prompt.jsonl"
    completed_by_id: Dict[str, Dict[str, Any]] = {}
    if not smoke_test and resume and log_path.exists():
        for row in read_jsonl(log_path):
            completed_by_id[row["id"]] = row
        print(f"[Task 2C] Resuming: {len(completed_by_id)}/{len(prepared_cases)} prompts already completed.", flush=True)
    elif log_path.exists():
        log_path.unlink()

    all_prompt_records: List[Dict[str, Any]] = []
    qualitative_samples: List[Dict[str, Any]] = []

    for idx, prep in enumerate(prepared_cases):
        pid = prep["id"]
        if pid in completed_by_id:
            all_prompt_records.append(completed_by_id[pid])
            continue

        L_ctx = prep["L_ctx"]
        L_q = prep["L_q"]
        pre_q_ids = torch.tensor([prep["pre_q_ids"]], dtype=torch.long, device=device)
        q_suffix_ids = torch.tensor([prep["q_suffix_ids"]], dtype=torch.long, device=device)
        ev_set = set(prep["evidence_tok_indices_in_ctx"])
        n_ev = max(len(ev_set), 1)

        # Case-specific matched byte budget = INT4 chanK/tokenV serialized bytes on pre-question cache
        full_ctx_bytes = fp16_kv_bytes(L_ctx, n_layers, n_kv_heads, head_dim)
        budget_ctx_bytes = quant_kv_bytes(L_ctx, 4, n_layers, n_kv_heads, head_dim)
        k_keep, evict_ctx_bytes = max_fp16_positions_for_budget(budget_ctx_bytes, L_ctx, n_layers, n_kv_heads, head_dim)
        hybrid_alloc = solve_hybrid_allocation(budget_ctx_bytes, L_ctx, n_layers, n_kv_heads, head_dim)

        # Shared pre-question prefill (measured once per prompt)
        clear_cuda()
        torch.cuda.synchronize()
        t_pf0 = time.perf_counter()
        with torch.no_grad():
            base_ctx_cache = model(input_ids=pre_q_ids, use_cache=True).past_key_values
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t_pf0) * 1000.0
        prefill_peak_bytes = int(torch.cuda.max_memory_allocated())

        # Compute question-aware importance scores once (used by C2_question_aware & C4_hybrid)
        importance, scoring_ms = compute_question_aware_scores(model, base_ctx_cache, q_suffix_ids, L_ctx)

        per_policy: Dict[str, Any] = {}

        # ---- 1. C1: Full FP16 Reference ----
        cache_c1 = clone_dynamic_cache(base_ctx_cache)
        r_c1 = run_policy_from_ctx_cache(
            model, tokenizer, cache_c1, q_suffix_ids, prep["ans_ids"], L_ctx,
        )
        ref_logprobs = r_c1.pop("post_q_logprobs")
        shared_dec_bytes_c1 = fp16_kv_bytes(L_q + r_c1["num_generated_tokens"], n_layers, n_kv_heads, head_dim)
        per_policy["C1_full_fp16"] = {
            **r_c1,
            "exact_match": int(r_c1["extracted_code"] == prep["code"]),
            "evidence_retained_frac": 1.0,
            "ctx_packed_bytes": full_ctx_bytes,
            "decode_extra_bytes": shared_dec_bytes_c1,
            "total_live_cache_bytes": full_ctx_bytes + shared_dec_bytes_c1,
            "realized_ctx_frac_of_fp16": 1.0,
            "scoring_ms": 0.0,
            "compress_ms": 0.0,
            "recon_error": None,
        }
        del cache_c1

        # ---- 2. C2(i): Sink + Recent FP16 Eviction ----
        cache_sr = clone_dynamic_cache(base_ctx_cache)
        torch.cuda.synchronize()
        tc0 = time.perf_counter()
        n_tail = k_keep - SINK_TOKENS
        keep_sr = torch.cat([
            torch.arange(0, SINK_TOKENS, device=device),
            torch.arange(L_ctx - n_tail, L_ctx, device=device),
        ])
        apply_eviction_inplace(cache_sr, keep_sr)
        torch.cuda.synchronize()
        c_sr_ms = (time.perf_counter() - tc0) * 1000.0
        r_sr = run_policy_from_ctx_cache(
            model, tokenizer, cache_sr, q_suffix_ids, prep["ans_ids"], L_ctx, ref_post_q_logprobs=ref_logprobs,
        )
        r_sr.pop("post_q_logprobs")
        keep_sr_set = set(keep_sr.tolist())
        ev_ret_sr = len(ev_set & keep_sr_set) / float(n_ev)
        dec_bytes_sr = fp16_kv_bytes(L_q + r_sr["num_generated_tokens"], n_layers, n_kv_heads, head_dim)
        per_policy["C2_sink_recent_fp16"] = {
            **r_sr,
            "exact_match": int(r_sr["extracted_code"] == prep["code"]),
            "evidence_retained_frac": ev_ret_sr,
            "kept_fp16_positions": k_keep,
            "ctx_packed_bytes": evict_ctx_bytes,
            "decode_extra_bytes": dec_bytes_sr,
            "total_live_cache_bytes": evict_ctx_bytes + dec_bytes_sr,
            "realized_ctx_frac_of_fp16": evict_ctx_bytes / float(full_ctx_bytes),
            "scoring_ms": 0.0,
            "compress_ms": c_sr_ms,
            "recon_error": None,
        }
        del cache_sr

        # ---- 3. C2(ii): Question-Aware FP16 Eviction ----
        cache_qa = clone_dynamic_cache(base_ctx_cache)
        torch.cuda.synchronize()
        tc0 = time.perf_counter()
        mid_scores = importance[SINK_TOKENS : L_ctx - RECENT_TOKENS]
        n_mid_keep = k_keep - SINK_TOKENS - RECENT_TOKENS
        if n_mid_keep > 0:
            top_mid = torch.topk(mid_scores, k=n_mid_keep, largest=True).indices + SINK_TOKENS
            keep_qa = torch.cat([
                torch.arange(0, SINK_TOKENS, device=device),
                top_mid,
                torch.arange(L_ctx - RECENT_TOKENS, L_ctx, device=device),
            ]).sort().values
        else:
            keep_qa = keep_sr
        apply_eviction_inplace(cache_qa, keep_qa)
        torch.cuda.synchronize()
        c_qa_ms = (time.perf_counter() - tc0) * 1000.0
        r_qa = run_policy_from_ctx_cache(
            model, tokenizer, cache_qa, q_suffix_ids, prep["ans_ids"], L_ctx, ref_post_q_logprobs=ref_logprobs,
        )
        r_qa.pop("post_q_logprobs")
        keep_qa_set = set(keep_qa.tolist())
        ev_ret_qa = len(ev_set & keep_qa_set) / float(n_ev)
        dec_bytes_qa = fp16_kv_bytes(L_q + r_qa["num_generated_tokens"], n_layers, n_kv_heads, head_dim)
        per_policy["C2_question_aware_fp16"] = {
            **r_qa,
            "exact_match": int(r_qa["extracted_code"] == prep["code"]),
            "evidence_retained_frac": ev_ret_qa,
            "kept_fp16_positions": k_keep,
            "ctx_packed_bytes": evict_ctx_bytes,
            "decode_extra_bytes": dec_bytes_qa,
            "total_live_cache_bytes": evict_ctx_bytes + dec_bytes_qa,
            "realized_ctx_frac_of_fp16": evict_ctx_bytes / float(full_ctx_bytes),
            "scoring_ms": scoring_ms,
            "compress_ms": c_qa_ms,
            "recon_error": None,
        }
        del cache_qa

        # ---- 4 & 5. C3: Uniform INT4 (tokenK/tokenV and chanK/tokenV) ----
        L_q_ctx = (L_ctx // GROUP_SIZE) * GROUP_SIZE
        for pname, layout in [
            ("C3_int4_tokenK_tokenV", "tokenK_tokenV"),
            ("C3_int4_chanK_tokenV", "chanK_tokenV"),
        ]:
            cache_q = clone_dynamic_cache(base_ctx_cache)
            torch.cuda.synchronize()
            tc0 = time.perf_counter()
            recon_err = quantize_kv_chunk_inplace(cache_q, 0, L_q_ctx, layout=layout, bits=4, g=GROUP_SIZE)
            torch.cuda.synchronize()
            c_q_ms = (time.perf_counter() - tc0) * 1000.0
            r_q = run_policy_from_ctx_cache(
                model, tokenizer, cache_q, q_suffix_ids, prep["ans_ids"], L_ctx,
                ref_post_q_logprobs=ref_logprobs,
                decode_quant_layout=layout, decode_quant_bits=4, quantized_prefix_len=L_q_ctx,
            )
            r_q.pop("post_q_logprobs")
            total_len_final = L_ctx + L_q + r_q["num_generated_tokens"]
            total_live_b = quant_kv_bytes(total_len_final, 4, n_layers, n_kv_heads, head_dim)
            per_policy[pname] = {
                **r_q,
                "exact_match": int(r_q["extracted_code"] == prep["code"]),
                "evidence_retained_frac": 1.0,  # full context retained at INT4
                "ctx_packed_bytes": budget_ctx_bytes,
                "decode_extra_bytes": total_live_b - budget_ctx_bytes,
                "total_live_cache_bytes": total_live_b,
                "realized_ctx_frac_of_fp16": budget_ctx_bytes / float(full_ctx_bytes),
                "scoring_ms": 0.0,
                "compress_ms": c_q_ms,
                "recon_error": recon_err,
            }
            del cache_q

        # ---- 6. C4: Hybrid (Selected FP16 + Channel-K/Token-V INT2) ----
        cache_hyb = clone_dynamic_cache(base_ctx_cache)
        torch.cuda.synchronize()
        tc0 = time.perf_counter()
        n_fp16_hyb = hybrid_alloc["num_fp16_positions"]
        n_mid_fp16 = n_fp16_hyb - SINK_TOKENS - RECENT_TOKENS
        mid_all = torch.arange(SINK_TOKENS, L_ctx - RECENT_TOKENS, device=device)
        if n_mid_fp16 > 0:
            top_mid_rel = torch.topk(mid_scores, k=n_mid_fp16, largest=True).indices
            fp16_mask = torch.zeros(L_ctx, dtype=torch.bool, device=device)
            fp16_mask[:SINK_TOKENS] = True
            fp16_mask[L_ctx - RECENT_TOKENS :] = True
            fp16_mask[mid_all[top_mid_rel]] = True
        else:
            fp16_mask = torch.zeros(L_ctx, dtype=torch.bool, device=device)
            fp16_mask[:SINK_TOKENS] = True
            fp16_mask[L_ctx - RECENT_TOKENS :] = True

        low_prec_idx = torch.nonzero(~fp16_mask, as_tuple=False).squeeze(-1)
        assert int(low_prec_idx.numel()) == hybrid_alloc["num_int2_positions"]
        recon_hyb = apply_hybrid_fp16_int2_inplace(cache_hyb, low_prec_idx, g=GROUP_SIZE)
        torch.cuda.synchronize()
        c_hyb_ms = (time.perf_counter() - tc0) * 1000.0

        r_hyb = run_policy_from_ctx_cache(
            model, tokenizer, cache_hyb, q_suffix_ids, prep["ans_ids"], L_ctx,
            ref_post_q_logprobs=ref_logprobs,
        )
        r_hyb.pop("post_q_logprobs")
        fp16_hyb_set = set(torch.nonzero(fp16_mask, as_tuple=False).squeeze(-1).tolist())
        ev_ret_hyb_fp16 = len(ev_set & fp16_hyb_set) / float(n_ev)
        dec_bytes_hyb = fp16_kv_bytes(L_q + r_hyb["num_generated_tokens"], n_layers, n_kv_heads, head_dim)
        per_policy["C4_hybrid_fp16_int2"] = {
            **r_hyb,
            "exact_match": int(r_hyb["extracted_code"] == prep["code"]),
            "evidence_retained_frac": ev_ret_hyb_fp16,  # fraction of evidence tokens retained in FP16 tier
            "hybrid_allocation": hybrid_alloc,
            "ctx_packed_bytes": hybrid_alloc["total_bytes"],
            "decode_extra_bytes": dec_bytes_hyb,
            "total_live_cache_bytes": hybrid_alloc["total_bytes"] + dec_bytes_hyb,
            "realized_ctx_frac_of_fp16": hybrid_alloc["total_bytes"] / float(full_ctx_bytes),
            "scoring_ms": scoring_ms,
            "compress_ms": c_hyb_ms,
            "recon_error": recon_hyb,
        }
        del cache_hyb, base_ctx_cache

        prompt_rec = {
            "id": pid,
            "gold_code": prep["code"],
            "target_tokens": prep["target_tokens"],
            "position": prep["position"],
            "distractors": prep["distractors"],
            "L_full": prep["L_full"],
            "L_ctx": L_ctx,
            "L_q": L_q,
            "evidence_tok_count_in_ctx": len(ev_set),
            "full_ctx_fp16_bytes": full_ctx_bytes,
            "matched_int4_budget_bytes": budget_ctx_bytes,
            "prefill_ms": prefill_ms,
            "prefill_peak_allocated_bytes": prefill_peak_bytes,
            "policies": per_policy,
        }
        all_prompt_records.append(prompt_rec)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(prompt_rec, sort_keys=True) + "\n")

        # Save detailed masks for first 6 diverse cases as aligned qualitative examples
        if len(qualitative_samples) < 6:
            qualitative_samples.append({
                "id": pid,
                "gold_code": prep["code"],
                "target_tokens": prep["target_tokens"],
                "position": prep["position"],
                "distractors": prep["distractors"],
                "L_ctx": L_ctx,
                "evidence_tok_indices_in_ctx": prep["evidence_tok_indices_in_ctx"],
                "ev_retained_mask_qa_fp16": [int(t in keep_qa_set) for t in prep["evidence_tok_indices_in_ctx"]],
                "ev_retained_mask_sr_fp16": [int(t in keep_sr_set) for t in prep["evidence_tok_indices_in_ctx"]],
                "ev_retained_mask_hybrid_fp16": [int(t in fp16_hyb_set) for t in prep["evidence_tok_indices_in_ctx"]],
                "generations": {p: per_policy[p]["generated_text"] for p in POLICY_NAMES},
                "extracted_codes": {p: per_policy[p]["extracted_code"] for p in POLICY_NAMES},
            })

        if (idx + 1) % 15 == 0 or (idx + 1) == len(prepared_cases):
            acc_str = " | ".join(
                f"{p[:7]}={100.0 * np.mean([r['policies'][p]['exact_match'] for r in all_prompt_records]):4.1f}%"
                for p in POLICY_NAMES
            )
            print(f"  [2C] {idx+1:3d}/{len(prepared_cases)} | {acc_str}", flush=True)

    # Free Qwen model before running independent B=8, L=4096 packed-cache microbenchmark
    del model
    clear_cuda()

    print("[Task 2C] Running C5 independent packed-cache encode/decode microbenchmark...", flush=True)
    microbench = run_packed_cache_microbenchmark(
        n_layers=n_layers, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        smoke_test=smoke_test, device=device,
    )

    # =========================================================================
    # Aggregate Summary, Factor Breakdowns, and Paired 95% Bootstrap CIs
    # =========================================================================
    ref_ok_mask = [r["policies"]["C1_full_fp16"]["exact_match"] == 1 for r in all_prompt_records]
    n_ref_ok = max(sum(ref_ok_mask), 1)

    overall_summary: Dict[str, Any] = {}
    for p in POLICY_NAMES:
        ems = [r["policies"][p]["exact_match"] for r in all_prompt_records]
        ems_cond = [r["policies"][p]["exact_match"] for r, ok in zip(all_prompt_records, ref_ok_mask) if ok]
        recons = [r["policies"][p]["recon_error"] for r in all_prompt_records if r["policies"][p]["recon_error"] is not None]
        overall_summary[p] = {
            "exact_match_acc": float(np.mean(ems)),
            "conditional_acc_given_fp16_correct": float(np.mean(ems_cond)) if ems_cond else None,
            "fp16_correct_count": int(sum(ref_ok_mask)),
            "mean_ref_answer_nll": float(np.mean([r["policies"][p]["ref_answer_nll"] for r in all_prompt_records])),
            "mean_post_prefill_kl": float(np.mean([r["policies"][p]["post_prefill_logit_kl"] for r in all_prompt_records])),
            "mean_evidence_retained_frac": float(np.mean([r["policies"][p]["evidence_retained_frac"] for r in all_prompt_records])),
            "mean_ctx_packed_bytes": float(np.mean([r["policies"][p]["ctx_packed_bytes"] for r in all_prompt_records])),
            "mean_total_live_cache_bytes": float(np.mean([r["policies"][p]["total_live_cache_bytes"] for r in all_prompt_records])),
            "mean_realized_ctx_frac_of_fp16": float(np.mean([r["policies"][p]["realized_ctx_frac_of_fp16"] for r in all_prompt_records])),
            "mean_prefill_ms": float(np.mean([r["prefill_ms"] for r in all_prompt_records])),
            "mean_scoring_ms": float(np.mean([r["policies"][p]["scoring_ms"] for r in all_prompt_records])),
            "mean_compress_ms": float(np.mean([r["policies"][p]["compress_ms"] for r in all_prompt_records])),
            "mean_q_suffix_ms": float(np.mean([r["policies"][p]["q_suffix_ms"] for r in all_prompt_records])),
            "mean_decode_ms": float(np.mean([r["policies"][p]["decode_ms"] for r in all_prompt_records])),
            "max_peak_allocated_bytes": int(max(r["policies"][p]["peak_allocated_bytes"] for r in all_prompt_records)),
            "mean_k_rel_l2": float(np.mean([x["k_rel_l2"] for x in recons])) if recons else None,
            "mean_v_rel_l2": float(np.mean([x["v_rel_l2"] for x in recons])) if recons else None,
            "mean_k_mse": float(np.mean([x["k_mse"] for x in recons])) if recons else None,
            "mean_v_mse": float(np.mean([x["v_mse"] for x in recons])) if recons else None,
        }

    def _group_acc(key_fn) -> Dict[str, Dict[str, Any]]:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for r in all_prompt_records:
            buckets.setdefault(str(key_fn(r)), []).append(r)
        out = {}
        for bk, rows in sorted(buckets.items()):
            out[bk] = {
                "count": len(rows),
                "mean_L_full": float(np.mean([x["L_full"] for x in rows])),
                **{p: float(np.mean([x["policies"][p]["exact_match"] for x in rows])) for p in POLICY_NAMES},
            }
        return out

    breakdowns = {
        "by_target_tokens": _group_acc(lambda r: r["target_tokens"]),
        "by_position": _group_acc(lambda r: r["position"]),
        "by_distractors": _group_acc(lambda r: r["distractors"]),
        "by_full_cell": _group_acc(lambda r: f"L{r['target_tokens']}_{r['position']}_d{r['distractors']}"),
    }

    # Paired bootstrap 95% CIs (2,000 resamples) over prompts for principal differences
    boot_pairs = [
        ("C2_question_aware_fp16", "C2_sink_recent_fp16"),
        ("C3_int4_chanK_tokenV", "C3_int4_tokenK_tokenV"),
        ("C3_int4_chanK_tokenV", "C2_question_aware_fp16"),
        ("C4_hybrid_fp16_int2", "C2_question_aware_fp16"),
        ("C4_hybrid_fp16_int2", "C3_int4_chanK_tokenV"),
        ("C4_hybrid_fp16_int2", "C1_full_fp16"),
        ("C3_int4_chanK_tokenV", "C1_full_fp16"),
    ]
    bootstrap_cis = {}
    for pa, pb in boot_pairs:
        va = [r["policies"][pa]["exact_match"] for r in all_prompt_records]
        vb = [r["policies"][pb]["exact_match"] for r in all_prompt_records]
        bootstrap_cis[f"{pa}_minus_{pb}"] = paired_bootstrap_mean_diff(va, vb, n_resamples=2000, seed=6304)

    payload = {
        "metadata": collect_run_metadata(
            data_root,
            extra={
                "task": "task2_part_c_kv_cache",
                "smoke_test": smoke_test,
                "model_id": model_id,
                "model_revision": model_rev,
                "num_prompts": len(all_prompt_records),
                "selection_scope": "globally_shared_across_heads_and_layers_int32_indices",
                "eviction_safeguards": {"sink_tokens": SINK_TOKENS, "recent_tokens": RECENT_TOKENS},
                "quant_convention": "asymmetric_group64_fp16scale_fp16offset_trailing_Lmod64_fp16",
            },
        ),
        "c1_validation": c1_validation,
        "overall_summary": overall_summary,
        "breakdowns": breakdowns,
        "paired_bootstrap_cis": bootstrap_cis,
        "packed_cache_microbenchmark": microbench,
        "qualitative_samples": qualitative_samples,
    }
    save_json(out_dir / "part_c_results.json", payload)
    print(f"[Task 2C] COMPLETE! Saved {out_dir / 'part_c_results.json'}", flush=True)
    return payload
