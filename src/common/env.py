from __future__ import annotations

import datetime
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


DEFAULT_SEED = 6304


def set_seed(seed: int = DEFAULT_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_git_commit(repo_dir: Optional[Path] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def load_source_revisions(data_root: Path) -> Dict[str, Any]:
    lock_path = Path(data_root) / "data" / "local_source_revisions.json"
    if not lock_path.exists():
        raise FileNotFoundError(
            f"Missing {lock_path}. Run `python scripts/setup_data.py --public` in atml-pa4-datasets first."
        )
    return json.loads(lock_path.read_text(encoding="utf-8"))


def collect_run_metadata(
    data_root: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import huggingface_hub
    import transformers

    try:
        import peft
        peft_ver = peft.__version__
    except ImportError:
        peft_ver = None

    meta: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "peft": peft_ver,
        "repo_commit": get_git_commit(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        meta["gpu_name"] = props.name
        meta["gpu_count"] = torch.cuda.device_count()
        meta["gpu_total_memory_bytes"] = props.total_memory
        meta["gpu_compute_capability"] = f"{props.major}.{props.minor}"

    if data_root is not None:
        lock_path = Path(data_root) / "data" / "local_source_revisions.json"
        if lock_path.exists():
            meta["source_revisions"] = json.loads(lock_path.read_text(encoding="utf-8"))

    if extra:
        meta.update(extra)
    return meta


def save_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def num_cache_layers(cache: Any) -> int:
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache.key_cache)


def get_kv_layer(cache: Any, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def set_kv_layer(cache: Any, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        layer.keys = keys
        layer.values = values
    else:
        cache.key_cache[layer_idx] = keys
        cache.value_cache[layer_idx] = values
