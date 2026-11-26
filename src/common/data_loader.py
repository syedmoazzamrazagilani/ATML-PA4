from __future__ import annotations

import gzip
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    rows: List[Dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterator[Dict[str, Any]] | List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n")


class PA4Data:
    """Unified access to all locked assets inside atml-pa4-datasets."""

    def __init__(self, data_root: Optional[str | Path] = None):
        root = data_root or os.environ.get("PA4_DATA_ROOT", "atml-pa4-datasets")
        self.root = Path(root).resolve()
        self.data_dir = self.root / "data"
        self.public_dir = self.data_dir / "public_locked"
        self.mqar_dir = self.data_dir / "mqar"
        self.kv_dir = self.data_dir / "kv_retrieval"
        self.short_dir = self.data_dir / "short_form"
        self.bank_dir = self.data_dir / "prefix_banks"

    def revisions(self) -> Dict[str, str]:
        lock = json.loads((self.data_dir / "local_source_revisions.json").read_text(encoding="utf-8"))
        return lock["revisions"]

    def model_spec(self, key: str) -> tuple[str, str]:
        """key in {'model_base', 'model_target', 'model_draft'} -> (repo_id, commit_sha)."""
        lock = json.loads((self.data_dir / "local_source_revisions.json").read_text(encoding="utf-8"))
        return lock["repos"][key], lock["revisions"][key]

    def c4_calibration(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.public_dir / "c4_calibration_locked.jsonl.gz")

    def c4_development(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.public_dir / "c4_development_locked.jsonl.gz")

    def wikitext2_test(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.public_dir / "wikitext2_test_windows_locked.jsonl.gz")

    def c4_validation(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.public_dir / "c4_validation_windows_locked.jsonl.gz")

    def hellaswag_val(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        rows = read_jsonl(self.public_dir / "hellaswag_val_locked.jsonl.gz")
        return rows[:limit] if limit is not None else rows

    def mqar_dev(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.mqar_dir / "mqar_development.jsonl.gz")

    def mqar_test(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.mqar_dir / "mqar_test.jsonl.gz")

    def load_mqar_generator(self):
        mod_path = self.root / "scripts" / "mqar_generator.py"
        spec = importlib.util.spec_from_file_location("pa4_mqar_generator", mod_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    def kv_retrieval_cases(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.kv_dir / "kv_retrieval_cases.jsonl.gz")

    def gsm8k_train(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.public_dir / "gsm8k_train_locked.jsonl.gz")

    def gsm8k_test(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.public_dir / "gsm8k_test_locked.jsonl.gz")

    def short_train(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.short_dir / "short_train.jsonl")

    def short_test(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.short_dir / "short_test.jsonl")

    def prefix_banks(self) -> List[Dict[str, Any]]:
        return read_jsonl(self.bank_dir / "prefix_banks_locked.jsonl.gz")
