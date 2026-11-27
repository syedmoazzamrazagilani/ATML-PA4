from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.data_loader import PA4Data
from src.common.env import collect_run_metadata, save_json, set_seed
from src.common.timing import clear_cuda, measure_cuda_fn


T_MQAR = 256
VOCAB_SIZE = 256
NUM_VALUE_CLASSES = 64  # Value token IDs 128..191
VALUE_OFFSET = 128
DISTANCE_BINS = [(0, 32), (32, 64), (64, 128), (128, 256)]


def make_sinusoidal_pos_emb(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # [1, max_len, d_model]


def build_window_causal_mask(T: int, window: int, device: torch.device) -> torch.Tensor:
    """Boolean mask [1, 1, T, T] where True means ALLOWED to attend: 0 <= i - j <= window - 1."""
    idx = torch.arange(T, device=device)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # [T, T]
    return ((diff >= 0) & (diff < window)).unsqueeze(0).unsqueeze(0)


def build_local_dilated_causal_mask(T: int, n_heads: int, device: torch.device) -> torch.Tensor:
    """
    Boolean mask [1, n_heads, T, T]:
      - First half of heads (2 heads): local preceding 64 positions (0 <= i - j <= 63)
      - Second half of heads (2 heads): 64 positions at dilated offsets {0, 4, 8, ..., 252}
    """
    idx = torch.arange(T, device=device)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)
    local_m = (diff >= 0) & (diff < 64)
    dilated_m = (diff >= 0) & (diff <= 252) & ((diff % 4) == 0)
    half = n_heads // 2
    masks = [local_m] * half + [dilated_m] * (n_heads - half)
    return torch.stack(masks, dim=0).unsqueeze(0)  # [1, H, T, T]


class MQARAttention(nn.Module):
    """
    Unified attention operator supporting:
      - 'dense': exact causal softmax attention
      - 'window64': causal sliding-window W=64
      - 'local_dilated': 2 local W=64 heads + 2 dilated stride-4 (64-pos) heads
      - 'linear': causal ELU(x)+1 kernel linear attention
      - 'selective_conv_linear': B3 intervention — causal depthwise short-conv on Q/K
        + input-dependent selective write gate g_t in (0, 1) with feature dim d_feat
        matched so total persistent state bytes per layer <= baseline linear attention.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        attn_type: str = "dense",
        window_size: int = 64,
        conv_kernel: int = 4,
        linear_feat_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_type = attn_type
        self.window_size = window_size
        self.conv_kernel = conv_kernel
        self.d_feat = linear_feat_dim if linear_feat_dim is not None else self.d_head

        if attn_type == "selective_conv_linear":
            # Feature-dim matched to keep state bytes <= baseline linear attention (4,224 floats/layer)
            self.q_proj = nn.Linear(d_model, n_heads * self.d_feat, bias=False)
            self.k_proj = nn.Linear(d_model, n_heads * self.d_feat, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)
            # Causal depthwise 1D convolution on Q and K inputs (binds k_j at 2j with v_j at 2j+1)
            self.conv_qk = nn.Conv1d(
                d_model, d_model, kernel_size=conv_kernel, groups=d_model, bias=True
            )
            # Per-head selective write gate g_t in (0, 1) to suppress filler tokens in recurrent memory
            self.write_gate = nn.Linear(d_model, n_heads, bias=True)
            nn.init.constant_(self.write_gate.bias, -1.0)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer("_cached_mask", torch.empty(0), persistent=False)
        self._cached_mask_len = 0

    def _get_mask(self, T: int, device: torch.device, override_mode: Optional[str]) -> Optional[torch.Tensor]:
        mode = override_mode or self.attn_type
        if mode == "dense":
            return None
        if override_mode is None and self._cached_mask_len == T and self._cached_mask.device == device:
            return self._cached_mask
        if mode == "window64":
            m = build_window_causal_mask(T, self.window_size, device)
        elif mode == "local_dilated":
            m = build_local_dilated_causal_mask(T, self.n_heads, device)
        else:
            return None
        if override_mode is None:
            self._cached_mask = m
            self._cached_mask_len = T
        return m

    def _linear_causal_chunked(
        self,
        phi_q: torch.Tensor,
        phi_k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int = 256,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Exact causal linear attention via chunked prefix accumulation:
        O(T * d_feat * d_h) compute and O(chunk_size * d_feat * d_h) peak memory even at T=8192.
        """
        B, H, T, df = phi_q.shape
        dv = v.shape[-1]
        if T <= chunk_size:
            kv = torch.einsum("bhtf,bhtv->bhtfv", phi_k, v)
            S = torch.cumsum(kv, dim=2)
            z = torch.cumsum(phi_k, dim=2)
            num = torch.einsum("bhtf,bhtfv->bhtv", phi_q, S)
            den = torch.einsum("bhtf,bhtf->bht", phi_q, z).unsqueeze(-1).clamp_min(eps)
            return num / den

        S_state = torch.zeros((B, H, df, dv), device=phi_q.device, dtype=phi_q.dtype)
        z_state = torch.zeros((B, H, df), device=phi_q.device, dtype=phi_q.dtype)
        outs = []
        for s in range(0, T, chunk_size):
            e = min(s + chunk_size, T)
            qc = phi_q[:, :, s:e, :]
            kc = phi_k[:, :, s:e, :]
            vc = v[:, :, s:e, :]

            kv_c = torch.einsum("bhtf,bhtv->bhtfv", kc, vc)
            S_local = torch.cumsum(kv_c, dim=2) + S_state.unsqueeze(2)
            z_local = torch.cumsum(kc, dim=2) + z_state.unsqueeze(2)

            num = torch.einsum("bhtf,bhtfv->bhtv", qc, S_local)
            den = torch.einsum("bhtf,bhtf->bht", qc, z_local).unsqueeze(-1).clamp_min(eps)
            outs.append(num / den)

            S_state = S_local[:, :, -1, :, :]
            z_state = z_local[:, :, -1, :]
        return torch.cat(outs, dim=2)

    def forward(self, x: torch.Tensor, eval_override: Optional[str] = None) -> torch.Tensor:
        B, T, C = x.shape
        mode = eval_override or self.attn_type

        if self.attn_type == "selective_conv_linear":
            # Causal depthwise conv over preceding conv_kernel tokens
            x_t = x.transpose(1, 2)  # [B, C, T]
            x_pad = F.pad(x_t, (self.conv_kernel - 1, 0))
            x_conv = F.silu(self.conv_qk(x_pad)).transpose(1, 2)  # [B, T, C]

            q = self.q_proj(x_conv).view(B, T, self.n_heads, self.d_feat).transpose(1, 2)
            k = self.k_proj(x_conv).view(B, T, self.n_heads, self.d_feat).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            gate = torch.sigmoid(self.write_gate(x)).transpose(1, 2).unsqueeze(-1)  # [B, H, T, 1]

            phi_q = F.elu(q) + 1.0
            phi_k = (F.elu(k) + 1.0) * gate
            y = self._linear_causal_chunked(phi_q, phi_k, v)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.o_proj(y)

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if mode == "linear":
            phi_q = F.elu(q) + 1.0
            phi_k = F.elu(k) + 1.0
            y = self._linear_causal_chunked(phi_q, phi_k, v)
        elif mode == "topk64":
            # Content-selected causal top-64 attention (post-hoc isolation control; quadratic score compute)
            scale = 1.0 / math.sqrt(self.d_head)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, float("-inf"))
            k_sel = min(64, T)
            top_vals, top_idx = torch.topk(scores, k=k_sel, dim=-1)
            masked = torch.full_like(scores, float("-inf"))
            masked.scatter_(-1, top_idx, top_vals)
            probs = F.softmax(masked, dim=-1, dtype=torch.float32).to(q.dtype)
            y = torch.matmul(probs, v)
        elif mode == "dense":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = self._get_mask(T, x.device, eval_override)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class MQARBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        attn_type: str = "dense",
        conv_kernel: int = 4,
        linear_feat_dim: Optional[int] = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MQARAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_type=attn_type,
            conv_kernel=conv_kernel,
            linear_feat_dim=linear_feat_dim,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, eval_override: Optional[str] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), eval_override=eval_override)
        x = x + self.mlp(self.ln2(x))
        return x


class MQARTransformer(nn.Module):
    """Fixed 2-layer pre-norm decoder-only Transformer for PA4 Task 2B."""

    def __init__(
        self,
        attn_type: str = "dense",
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = T_MQAR,
        conv_kernel: int = 4,
        linear_feat_dim: Optional[int] = None,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_emb", make_sinusoidal_pos_emb(max_len, d_model), persistent=False)
        self.blocks = nn.ModuleList([
            MQARBlock(
                d_model=d_model,
                n_heads=n_heads,
                attn_type=attn_type,
                conv_kernel=conv_kernel,
                linear_feat_dim=linear_feat_dim,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, NUM_VALUE_CLASSES)

    def forward(
        self,
        tokens: torch.Tensor,
        query_positions: torch.Tensor,
        eval_override: Optional[str] = None,
    ) -> torch.Tensor:
        B, T = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            x = blk(x, eval_override=eval_override)
        x = self.ln_f(x)
        # Gather representations strictly at the K marked query positions
        idx = query_positions.unsqueeze(-1).expand(-1, -1, self.d_model)
        q_hidden = torch.gather(x, dim=1, index=idx)  # [B, K, d_model]
        return self.head(q_hidden)  # [B, K, 64]

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pack_mqar_rows(rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor([r["tokens"] for r in rows], dtype=torch.long),
        "qpos": torch.tensor([r["query_positions"] for r in rows], dtype=torch.long),
        "ans": torch.tensor([[a - VALUE_OFFSET for a in r["answer_ids"]] for r in rows], dtype=torch.long),
        "dist": torch.tensor([r["distances"] for r in rows], dtype=torch.long),
    }


def group_mqar_split(rows: List[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[str, torch.Tensor]]:
    grouped_raw: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (int(r["K"]), str(r["regime"]))
        grouped_raw.setdefault(key, []).append(r)
    return {k: pack_mqar_rows(v) for k, v in grouped_raw.items()}


def pregenerate_train_pool(
    mqar_mod,
    k: int,
    regime: str,
    num_examples: int,
) -> Dict[str, torch.Tensor]:
    """Generate deterministic training examples via the official mqar_generator.make_example."""
    t0 = time.perf_counter()
    tokens = np.empty((num_examples, T_MQAR), dtype=np.int16)
    qpos = np.empty((num_examples, k), dtype=np.int16)
    ans = np.empty((num_examples, k), dtype=np.int16)
    for i in range(num_examples):
        ex = mqar_mod.make_example("train", k, regime, i)
        tokens[i] = ex["tokens"]
        qpos[i] = ex["query_positions"]
        ans[i] = [a - VALUE_OFFSET for a in ex["answer_ids"]]
    dt = time.perf_counter() - t0
    print(f"  [2B] Pre-generated {num_examples:,} train examples for (K={k}, {regime}) in {dt:.1f}s", flush=True)
    return {
        "tokens": torch.from_numpy(tokens),
        "qpos": torch.from_numpy(qpos),
        "ans": torch.from_numpy(ans),
    }


@torch.no_grad()
def evaluate_mqar_grouped(
    model: MQARTransformer,
    grouped_data: Dict[Tuple[int, str], Dict[str, torch.Tensor]],
    eval_override: Optional[str] = None,
    batch_size: int = 128,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate accuracy and CE loss per (K, regime) and across finer distance bins."""
    model.eval()
    by_cell: Dict[str, Any] = {}
    bin_correct = {f"[{lo},{hi})": 0 for lo, hi in DISTANCE_BINS}
    bin_total = {f"[{lo},{hi})": 0 for lo, hi in DISTANCE_BINS}

    for (k, regime), pack in sorted(grouped_data.items()):
        toks = pack["tokens"]
        qpos = pack["qpos"]
        ans = pack["ans"]
        dist = pack["dist"]
        N = toks.shape[0]

        tot_correct = 0
        tot_queries = 0
        tot_loss = 0.0
        cell_bins_c = {f"[{lo},{hi})": 0 for lo, hi in DISTANCE_BINS}
        cell_bins_n = {f"[{lo},{hi})": 0 for lo, hi in DISTANCE_BINS}

        for s in range(0, N, batch_size):
            bt = toks[s : s + batch_size].to(device=device, dtype=torch.long)
            bq = qpos[s : s + batch_size].to(device=device, dtype=torch.long)
            ba = ans[s : s + batch_size].to(device=device, dtype=torch.long)
            bd = dist[s : s + batch_size].to(device=device)

            logits = model(bt, bq, eval_override=eval_override)
            loss = F.cross_entropy(logits.reshape(-1, NUM_VALUE_CLASSES), ba.reshape(-1), reduction="sum")
            preds = logits.argmax(dim=-1)
            correct = (preds == ba)

            tot_loss += float(loss.item())
            tot_correct += int(correct.sum().item())
            tot_queries += int(ba.numel())

            for lo, hi in DISTANCE_BINS:
                bkey = f"[{lo},{hi})"
                m = (bd >= lo) & (bd < hi)
                cnt = int(m.sum().item())
                if cnt > 0:
                    c_cnt = int(correct[m].sum().item())
                    cell_bins_c[bkey] += c_cnt
                    cell_bins_n[bkey] += cnt
                    bin_correct[bkey] += c_cnt
                    bin_total[bkey] += cnt

        cell_key = f"K{k}_{regime}"
        by_cell[cell_key] = {
            "K": k,
            "regime": regime,
            "num_sequences": N,
            "num_queries": tot_queries,
            "accuracy": tot_correct / max(tot_queries, 1),
            "mean_loss": tot_loss / max(tot_queries, 1),
            "chance_label_space": 1.0 / NUM_VALUE_CLASSES,
            "chance_in_example": 1.0 / float(k),
            "distance_bins": {
                bkey: {
                    "accuracy": (cell_bins_c[bkey] / cell_bins_n[bkey]) if cell_bins_n[bkey] > 0 else None,
                    "count": cell_bins_n[bkey],
                }
                for bkey in cell_bins_c
            },
        }

    overall_bins = {
        bkey: {
            "accuracy": (bin_correct[bkey] / bin_total[bkey]) if bin_total[bkey] > 0 else None,
            "count": bin_total[bkey],
        }
        for bkey in bin_correct
    }
    return {"by_cell": by_cell, "overall_distance_bins": overall_bins}


def train_single_mqar_run(
    config_name: str,
    attn_type: str,
    train_k: int,
    train_regime: str,
    seed: int,
    train_pool: Dict[str, torch.Tensor],
    dev_grouped: Dict[Tuple[int, str], Dict[str, torch.Tensor]],
    test_grouped: Dict[Tuple[int, str], Dict[str, torch.Tensor]],
    num_updates: int = 4500,
    batch_size: int = 32,
    eval_steps: Tuple[int, ...] = (500, 1000, 1500, 2000, 3000, 4500),
    conv_kernel: int = 4,
    linear_feat_dim: Optional[int] = None,
    run_isolation_controls: bool = False,
    device: str = "cuda",
) -> Dict[str, Any]:
    set_seed(seed)
    model = MQARTransformer(
        attn_type=attn_type,
        d_model=128,
        n_heads=4,
        n_layers=2,
        vocab_size=VOCAB_SIZE,
        max_len=T_MQAR,
        conv_kernel=conv_kernel,
        linear_feat_dim=linear_feat_dim,
    ).to(device)

    peak_lr = 8e-4
    min_lr_ratio = 0.15
    warmup_steps = min(200, max(1, num_updates // 10))

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(num_updates - warmup_steps, 1))
        return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * prog)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Seed-specific permutation over the deterministic training pool
    n_pool = train_pool["tokens"].shape[0]
    g = torch.Generator().manual_seed(seed * 1009 + train_k * 31)
    perm = torch.randperm(n_pool, generator=g)

    # Subset of dev for quick learning-curve tracking (only the trained K across near/far/mixed)
    dev_for_k = {key: val for key, val in dev_grouped.items() if key[0] == train_k}

    curve: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    running_loss = 0.0
    loss_count = 0

    model.train()
    for step in range(1, num_updates + 1):
        start_i = ((step - 1) * batch_size) % n_pool
        if start_i + batch_size <= n_pool:
            b_idx = perm[start_i : start_i + batch_size]
        else:
            b_idx = torch.cat([perm[start_i:], perm[: (start_i + batch_size) % n_pool]])

        bt = train_pool["tokens"][b_idx].to(device=device, dtype=torch.long)
        bq = train_pool["qpos"][b_idx].to(device=device, dtype=torch.long)
        ba = train_pool["ans"][b_idx].to(device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(bt, bq)
        loss = F.cross_entropy(logits.reshape(-1, NUM_VALUE_CLASSES), ba.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += float(loss.item())
        loss_count += 1

        if step in eval_steps or step == num_updates:
            dev_res = evaluate_mqar_grouped(model, dev_for_k, device=device)
            curve.append({
                "step": step,
                "train_loss_avg": running_loss / max(loss_count, 1),
                "lr": float(scheduler.get_last_lr()[0]),
                "dev_near_acc": dev_res["by_cell"][f"K{train_k}_near"]["accuracy"],
                "dev_far_acc": dev_res["by_cell"][f"K{train_k}_far"]["accuracy"],
                "dev_mixed_acc": dev_res["by_cell"][f"K{train_k}_mixed"]["accuracy"],
                "dev_mixed_loss": dev_res["by_cell"][f"K{train_k}_mixed"]["mean_loss"],
            })
            running_loss = 0.0
            loss_count = 0
            model.train()

    train_time_s = time.perf_counter() - t_start

    # Final evaluation on frozen test split (all K and regimes for full transfer + matched K diagnostics)
    test_eval = evaluate_mqar_grouped(model, test_grouped, device=device)

    isolation = {}
    if run_isolation_controls:
        isolation["test_mask_window64"] = evaluate_mqar_grouped(
            model, test_grouped, eval_override="window64", device=device
        )
        isolation["test_mask_topk64"] = evaluate_mqar_grouped(
            model, test_grouped, eval_override="topk64", device=device
        )

    matched_near = test_eval["by_cell"][f"K{train_k}_near"]["accuracy"]
    matched_far = test_eval["by_cell"][f"K{train_k}_far"]["accuracy"]
    matched_mixed = test_eval["by_cell"][f"K{train_k}_mixed"]["accuracy"]
    print(
        f"  [2B] {config_name:32s} seed={seed} | {train_time_s:5.1f}s | "
        f"test(K={train_k:2d}) near={matched_near*100:5.1f}% far={matched_far*100:5.1f}% mixed={matched_mixed*100:5.1f}%",
        flush=True,
    )

    return {
        "config_name": config_name,
        "attn_type": attn_type,
        "train_K": train_k,
        "train_regime": train_regime,
        "seed": seed,
        "conv_kernel": conv_kernel if attn_type == "selective_conv_linear" else None,
        "linear_feat_dim": linear_feat_dim if attn_type == "selective_conv_linear" else None,
        "trainable_params": model.count_trainable_params(),
        "num_updates": num_updates,
        "train_time_s": train_time_s,
        "learning_curve": curve,
        "test_eval": test_eval,
        "isolation_controls": isolation,
    }


def compute_theoretical_costs_t256(b3_conv_kernel: int, b3_feat_dim: int) -> Dict[str, Any]:
    """Theoretical attention pair counts and persistent state bytes per layer at T=256 (FP32 floats)."""
    T = 256
    H = 4
    d_h = 32
    d_model = 128

    dense_pairs = H * (T * (T + 1) // 2)
    win_pairs_per_head = sum(min(i + 1, 64) for i in range(T))
    window_pairs = H * win_pairs_per_head
    dil_pairs_per_head = sum((i // 4) + 1 for i in range(T))  # offsets 0,4,8,...,<=i (max 64)
    local_dilated_pairs = 2 * win_pairs_per_head + 2 * dil_pairs_per_head

    # Persistent state per layer (for 1 sequence, FP32 = 4 bytes/float)
    kv_dense_bytes = 2 * T * H * d_h * 4
    kv_window_bytes = 2 * 64 * H * d_h * 4
    linear_state_floats = H * (d_h * d_h + d_h)  # S [H, 32, 32] + z [H, 32] = 4,224 floats
    linear_state_bytes = linear_state_floats * 4

    b3_rec_floats = H * (b3_feat_dim * d_h + b3_feat_dim)
    b3_conv_floats = (b3_conv_kernel - 1) * d_model
    b3_total_floats = b3_rec_floats + b3_conv_floats
    b3_total_bytes = b3_total_floats * 4

    return {
        "T": T, "H": H, "d_h": d_h, "bytes_per_float": 4,
        "dense": {"attended_pairs_total_4heads": dense_pairs, "kv_cache_bytes_per_layer": kv_dense_bytes},
        "window64": {
            "attended_pairs_total_4heads": window_pairs,
            "kv_buffer_bytes_per_layer": kv_window_bytes,
            "two_layer_max_receptive_field": 2 * (64 - 1),
        },
        "local_dilated": {
            "attended_pairs_total_4heads": local_dilated_pairs,
            "note": "2 local heads (W=64) + 2 dilated heads (stride-4 offsets 0..252); dense boolean mask used in PyTorch correctness impl.",
        },
        "linear_elu1": {
            "attended_pairs_total_4heads": None,
            "persistent_state_floats_per_layer": linear_state_floats,
            "persistent_state_bytes_per_layer": linear_state_bytes,
        },
        "b3_selective_conv_linear": {
            "conv_kernel": b3_conv_kernel,
            "linear_feat_dim": b3_feat_dim,
            "recurrent_state_floats_per_layer": b3_rec_floats,
            "conv_buffer_floats_per_layer": b3_conv_floats,
            "persistent_state_floats_per_layer": b3_total_floats,
            "persistent_state_bytes_per_layer": b3_total_bytes,
            "matched_le_linear_baseline": b3_total_bytes <= linear_state_bytes,
        },
    }


@torch.no_grad()
def run_inference_length_sweep(
    b3_conv_kernel: int,
    b3_feat_dim: int,
    smoke_test: bool = False,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Separate forward-only inference sweep at T in {256, 1024, 4096, 8192}
    with fixed random embeddings and matched B=2, H=4, d_h=32, isolating kernel execution from learnability.
    """
    B, H, d_h = 2, 4, 32
    d_model = H * d_h
    lengths = [256, 1024] if smoke_test else [256, 1024, 4096, 8192]
    reps = 2 if smoke_test else 8
    warmup = 1 if smoke_test else 3

    specs = [
        ("dense", "dense", None, "sdpa_causal_no_dense_mask"),
        ("window64", "window64", None, "explicit_dense_TxT_boolean_mask_in_sdpa"),
        ("local_dilated", "local_dilated", None, "explicit_dense_HxTxT_boolean_mask_in_sdpa"),
        ("linear_elu1", "linear", None, "chunked_256_causal_prefix_state_accumulation"),
        ("b3_selective_conv_linear", "selective_conv_linear", None, "depthwise_conv_plus_chunked_256_prefix_state"),
        ("topk64_posthoc", "dense", "topk64", "explicit_quadratic_TxT_scores_then_topk64"),
    ]

    results = []
    for L in lengths:
        gen = torch.Generator(device=device).manual_seed(6304 + L)
        x = torch.randn(B, L, d_model, generator=gen, device=device, dtype=torch.float32)
        for label, attn_type, override, impl_note in specs:
            clear_cuda()
            mod = MQARAttention(
                d_model=d_model,
                n_heads=H,
                attn_type=attn_type,
                conv_kernel=b3_conv_kernel,
                linear_feat_dim=b3_feat_dim,
            ).to(device).eval()
            try:
                stats = measure_cuda_fn(
                    lambda: mod(x, eval_override=override),
                    warmup=warmup, reps=reps, reset_peak_per_rep=True,
                )
                rec = {"T": L, "method": label, "implementation": impl_note, "status": "OK", **stats}
            except Exception as exc:
                clear_cuda()
                rec = {"T": L, "method": label, "implementation": impl_note, "status": f"ERR: {str(exc)[:160]}"}
            results.append(rec)
            ms = rec.get("mean_ms", float("nan"))
            mb = rec.get("peak_allocated_bytes", 0) / 1e6
            print(f"  [2B sweep] T={L:4d} {label:26s} | {rec['status']:4s} | {ms:7.2f}ms | peak={mb:6.1f}MB", flush=True)
            del mod
        del x
        clear_cuda()
    return results


def run_part_b(
    data_root: Path,
    out_dir: Path,
    smoke_test: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    set_seed(6304)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "part_b_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    data = PA4Data(data_root)
    mqar_mod = data.load_mqar_generator()

    # Verify generator invariants before training
    for regime in ("near", "far", "mixed"):
        for k in (4, 8, 16):
            ex = mqar_mod.make_example("train", k, regime, 0)
            mqar_mod.validate(ex)
    print("[Task 2B] MQAR generator invariant checks passed.", flush=True)

    dev_rows = data.mqar_dev()
    test_rows = data.mqar_test()
    if smoke_test:
        # Keep 16 examples per (K, regime) cell for fast smoke evaluation
        def _slice_cells(rows, per_cell=16):
            counts: Dict[Tuple[int, str], int] = {}
            out = []
            for r in rows:
                key = (int(r["K"]), str(r["regime"]))
                if counts.get(key, 0) < per_cell:
                    counts[key] = counts.get(key, 0) + 1
                    out.append(r)
            return out
        dev_rows = _slice_cells(dev_rows, 16)
        test_rows = _slice_cells(test_rows, 16)

    dev_grouped = group_mqar_split(dev_rows)
    test_grouped = group_mqar_split(test_rows)

    seeds = [6304] if smoke_test else [6304, 6305, 6306]
    num_updates = 20 if smoke_test else 4500
    batch_size = 32
    eval_steps = (10, 20) if smoke_test else (500, 1000, 1500, 2000, 3000, 4500)
    pool_size = num_updates * batch_size

    # Pre-generate deterministic train pools once for the 6 (K, regime) combinations used
    required_pools = [(4, "mixed"), (8, "mixed"), (16, "mixed"), (4, "near"), (8, "near"), (16, "near")]
    train_pools: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
    for k, reg in required_pools:
        train_pools[(k, reg)] = pregenerate_train_pool(mqar_mod, k, reg, pool_size)

    # All required B1 & B2 training configurations
    base_configs = [
        # B1: Dense baselines
        ("B1_dense_mixed_K4", "dense", 4, "mixed", True),
        ("B1_dense_mixed_K8", "dense", 8, "mixed", True),
        ("B1_dense_mixed_K16", "dense", 16, "mixed", True),
        ("B1_dense_near_K16", "dense", 16, "near", True),
        # B1: Sliding window W=64 baselines
        ("B1_window64_near_K4", "window64", 4, "near", False),
        ("B1_window64_near_K8", "window64", 8, "near", False),
        ("B1_window64_near_K16", "window64", 16, "near", False),
        ("B1_window64_mixed_K16", "window64", 16, "mixed", False),
        # B2: Linear ELU+1 and Local/Dilated sparse
        ("B2_linear_mixed_K4", "linear", 4, "mixed", False),
        ("B2_linear_mixed_K8", "linear", 8, "mixed", False),
        ("B2_linear_mixed_K16", "linear", 16, "mixed", False),
        ("B2_local_dilated_mixed_K4", "local_dilated", 4, "mixed", False),
        ("B2_local_dilated_mixed_K16", "local_dilated", 16, "mixed", False),
    ]

    all_runs: List[Dict[str, Any]] = []
    for cname, attn_type, k, reg, do_iso in base_configs:
        for seed in seeds:
            rpath = runs_dir / f"{cname}_seed{seed}.json"
            if not smoke_test and resume and rpath.exists():
                rec = json.loads(rpath.read_text(encoding="utf-8"))
                print(f"  [2B] Loaded cached {rpath.name}", flush=True)
            else:
                rec = train_single_mqar_run(
                    cname, attn_type, k, reg, seed,
                    train_pools[(k, reg)], dev_grouped, test_grouped,
                    num_updates=num_updates, batch_size=batch_size, eval_steps=eval_steps,
                    run_isolation_controls=do_iso,
                )
                save_json(rpath, rec)
            all_runs.append(rec)

    # =========================================================================
    # B3: Controlled Intervention — Selective-Gate + Depthwise-Conv Linear Attention
    # Dev-set selection on mqar_development.jsonl.gz (seed 6304, 1500 updates)
    # comparing conv_kernel=2 (d_feat=31, state=4220 floats) vs conv_kernel=4 (d_feat=29, state=4212 floats)
    # =========================================================================
    b3_sel_path = out_dir / "part_b3_dev_selection.json"
    baseline_params = MQARTransformer(attn_type="linear").count_trainable_params()

    if not smoke_test and resume and b3_sel_path.exists():
        b3_sel = json.loads(b3_sel_path.read_text(encoding="utf-8"))
        best_ck = int(b3_sel["selected"]["conv_kernel"])
        best_df = int(b3_sel["selected"]["linear_feat_dim"])
    else:
        print("[Task 2B] Running B3 dev-set candidate selection on mqar_development...", flush=True)
        dev_steps = 20 if smoke_test else 1500
        b3_cands = []
        best_cand = None
        for ck, df in [(2, 31), (4, 29)]:
            m_tmp = MQARTransformer(attn_type="selective_conv_linear", conv_kernel=ck, linear_feat_dim=df)
            p_cnt = m_tmp.count_trainable_params()
            rel_diff = abs(p_cnt - baseline_params) / float(baseline_params)
            assert rel_diff <= 0.05, f"B3 param count diff {rel_diff:.3%} > 5%"
            # Verify state bytes <= baseline linear attention (4,224 floats/layer)
            st_floats = 4 * (df * 32 + df) + (ck - 1) * 128
            assert st_floats <= 4224, f"B3 state floats {st_floats} > 4224"

            rec_dev = train_single_mqar_run(
                f"B3_dev_ck{ck}_df{df}", "selective_conv_linear", 16, "mixed", 6304,
                train_pools[(16, "mixed")], dev_grouped, dev_grouped,
                num_updates=dev_steps, batch_size=batch_size,
                eval_steps=(dev_steps,), conv_kernel=ck, linear_feat_dim=df,
            )
            dev_mixed_acc = rec_dev["learning_curve"][-1]["dev_mixed_acc"]
            cand = {
                "conv_kernel": ck,
                "linear_feat_dim": df,
                "trainable_params": p_cnt,
                "param_rel_diff_vs_linear": rel_diff,
                "state_floats_per_layer": st_floats,
                "dev_steps": dev_steps,
                "dev_mixed_acc": dev_mixed_acc,
                "dev_near_acc": rec_dev["learning_curve"][-1]["dev_near_acc"],
                "dev_far_acc": rec_dev["learning_curve"][-1]["dev_far_acc"],
            }
            b3_cands.append(cand)
            if best_cand is None or dev_mixed_acc > best_cand["dev_mixed_acc"]:
                best_cand = cand
        assert best_cand is not None
        best_ck = int(best_cand["conv_kernel"])
        best_df = int(best_cand["linear_feat_dim"])
        b3_sel = {
            "pre_evaluation_prediction": (
                "Adding a causal depthwise short convolution on Q/K (to bind adjacent key-value pairs k_j, v_j in Layer 1) "
                "and an input-dependent write gate g_t (to suppress filler tokens from polluting the fixed-size associative matrix S_t) "
                "will substantially improve K=16 near and far recall over ELU+1 linear attention at matched persistent state bytes."
            ),
            "selection_rule": "maximize_K16_mixed_accuracy_on_mqar_development_at_1500_updates_seed_6304",
            "baseline_linear_trainable_params": baseline_params,
            "candidates": b3_cands,
            "selected": best_cand,
        }
        save_json(b3_sel_path, b3_sel)

    # Train selected B3 intervention for full 4,500 updates across all 3 seeds
    b3_name = f"B3_selective_conv_linear_mixed_K16"
    for seed in seeds:
        rpath = runs_dir / f"{b3_name}_seed{seed}.json"
        if not smoke_test and resume and rpath.exists():
            rec = json.loads(rpath.read_text(encoding="utf-8"))
            print(f"  [2B] Loaded cached {rpath.name}", flush=True)
        else:
            rec = train_single_mqar_run(
                b3_name, "selective_conv_linear", 16, "mixed", seed,
                train_pools[(16, "mixed")], dev_grouped, test_grouped,
                num_updates=num_updates, batch_size=batch_size, eval_steps=eval_steps,
                conv_kernel=best_ck, linear_feat_dim=best_df,
            )
            save_json(rpath, rec)
        all_runs.append(rec)

    # Aggregate mean +/- std across seeds per config
    agg: Dict[str, Any] = {}
    by_cfg: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_runs:
        by_cfg.setdefault(r["config_name"], []).append(r)

    for cname, rlist in by_cfg.items():
        tk = rlist[0]["train_K"]
        near_vals = [r["test_eval"]["by_cell"][f"K{tk}_near"]["accuracy"] for r in rlist]
        far_vals = [r["test_eval"]["by_cell"][f"K{tk}_far"]["accuracy"] for r in rlist]
        mix_vals = [r["test_eval"]["by_cell"][f"K{tk}_mixed"]["accuracy"] for r in rlist]
        agg[cname] = {
            "attn_type": rlist[0]["attn_type"],
            "train_K": tk,
            "train_regime": rlist[0]["train_regime"],
            "trainable_params": rlist[0]["trainable_params"],
            "seeds": [r["seed"] for r in rlist],
            "matched_K_near_mean": float(np.mean(near_vals)),
            "matched_K_near_std": float(np.std(near_vals, ddof=1)) if len(near_vals) > 1 else 0.0,
            "matched_K_far_mean": float(np.mean(far_vals)),
            "matched_K_far_std": float(np.std(far_vals, ddof=1)) if len(far_vals) > 1 else 0.0,
            "matched_K_mixed_mean": float(np.mean(mix_vals)),
            "matched_K_mixed_std": float(np.std(mix_vals, ddof=1)) if len(mix_vals) > 1 else 0.0,
            "mean_train_time_s": float(np.mean([r["train_time_s"] for r in rlist])),
        }

    # Theoretical costs + Forward-only inference length sweep
    theo = compute_theoretical_costs_t256(best_ck, best_df)
    print("[Task 2B] Running forward-only inference length sweep (T=256..8192)...", flush=True)
    sweep = run_inference_length_sweep(best_ck, best_df, smoke_test=smoke_test)

    payload = {
        "metadata": collect_run_metadata(data_root, extra={"task": "task2_part_b_mqar", "smoke_test": smoke_test}),
        "b3_dev_selection": b3_sel,
        "theoretical_costs_t256": theo,
        "aggregated_summary": agg,
        "inference_length_sweep": sweep,
        "individual_run_files": sorted(p.name for p in runs_dir.glob("*.json")),
    }
    save_json(out_dir / "part_b_results.json", payload)
    print(f"[Task 2B] Saved {out_dir / 'part_b_results.json'}", flush=True)
    return payload
