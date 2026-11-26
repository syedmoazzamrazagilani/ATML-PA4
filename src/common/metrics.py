from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def paired_bootstrap_mean_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 6304,
) -> Dict[str, float]:
    """Paired bootstrap CI for mean(values_a - values_b)."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 0
    diffs = a - b
    n = len(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = diffs[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff_mean": float(diffs.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(n),
        "n_resamples": int(n_resamples),
    }


def paired_bootstrap_ppl_diff(
    nll_sums_a: Sequence[float],
    nll_sums_b: Sequence[float],
    scored_tokens: Sequence[int],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 6304,
) -> Dict[str, float]:
    """
    Paired window-level bootstrap for PPL_a - PPL_b and NLL_a - NLL_b.
    Resamples whole fixed evaluation windows and recomputes pooled token NLL
    before exponentiating, per PA4 Task 1 specification.
    """
    sa = np.asarray(nll_sums_a, dtype=np.float64)
    sb = np.asarray(nll_sums_b, dtype=np.float64)
    toks = np.asarray(scored_tokens, dtype=np.float64)
    assert sa.shape == sb.shape == toks.shape and sa.ndim == 1 and len(sa) > 0

    total_toks = toks.sum()
    nll_a = float(sa.sum() / total_toks)
    nll_b = float(sb.sum() / total_toks)
    ppl_a = float(np.exp(nll_a))
    ppl_b = float(np.exp(nll_b))

    n = len(sa)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_toks = toks[idx].sum(axis=1)
    boot_nll_a = sa[idx].sum(axis=1) / boot_toks
    boot_nll_b = sb[idx].sum(axis=1) / boot_toks
    boot_nll_diff = boot_nll_a - boot_nll_b
    boot_ppl_diff = np.exp(boot_nll_a) - np.exp(boot_nll_b)

    return {
        "nll_a": nll_a,
        "nll_b": nll_b,
        "nll_diff": float(nll_a - nll_b),
        "nll_diff_ci_low": float(np.quantile(boot_nll_diff, alpha / 2.0)),
        "nll_diff_ci_high": float(np.quantile(boot_nll_diff, 1.0 - alpha / 2.0)),
        "ppl_a": ppl_a,
        "ppl_b": ppl_b,
        "ppl_diff": float(ppl_a - ppl_b),
        "ppl_diff_ci_low": float(np.quantile(boot_ppl_diff, alpha / 2.0)),
        "ppl_diff_ci_high": float(np.quantile(boot_ppl_diff, 1.0 - alpha / 2.0)),
        "n_windows": int(n),
        "total_scored_tokens": int(total_toks),
        "n_resamples": int(n_resamples),
    }
