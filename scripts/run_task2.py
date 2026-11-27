from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ATML PA4 Task 2: Attention and Context Memory")
    p.add_argument("--data-root", type=str, default="atml-pa4-datasets")
    p.add_argument("--out-dir", type=str, default="results/task2")
    p.add_argument("--parts", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    p.add_argument("--smoke-test", action="store_true", help="Fast end-to-end verification of all selected parts")
    p.add_argument("--resume", action="store_true", help="Skip cached runs where supported")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "A" in args.parts:
        from src.task2.part_a_attention import run_part_a
        run_part_a(data_root=data_root, out_dir=out_dir, smoke_test=args.smoke_test)

    if "B" in args.parts:
        from src.task2.part_b_mqar import run_part_b
        run_part_b(data_root=data_root, out_dir=out_dir, smoke_test=args.smoke_test, resume=args.resume)

    if "C" in args.parts:
        from src.task2.part_c_kvcache import run_part_c
        run_part_c(data_root=data_root, out_dir=out_dir, smoke_test=args.smoke_test, resume=args.resume)

    tag = "SMOKE TEST PASSED" if args.smoke_test else "COMPLETE"
    print(f"\n[Task 2] {tag} for parts {args.parts}!", flush=True)


if __name__ == "__main__":
    main()
