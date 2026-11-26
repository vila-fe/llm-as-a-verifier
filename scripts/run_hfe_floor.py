#!/usr/bin/env python3
"""Run the HFE floor-reliability suite in HOOTL mode (Sofia / MacQAS Atómico).

Simulates `--runs` deterministic model runs per prompt from a per-capability
reliability profile, reports pass@1 and the strict family pass measure, checks
the canonical floor threshold against the Wilson 95% lower bound, and appends
the outcome to the Sofia policy registry log.

No API keys, models or hardware are needed. Once real trajectories exist, the
same suite can be scored with `python scripts/run.py hfe_floor`.

Usage:
    python scripts/run_hfe_floor.py
    python scripts/run_hfe_floor.py --seed 1 --runs 4 --threshold 0.85
    python scripts/run_hfe_floor.py --suite data/hfe_floor_trajs/hfe_floor_seed.jsonl
"""

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from llm_verifier.hfe_floor import DEFAULT_PROFILE, run_hootl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", default=os.path.join(ROOT_DIR, "data", "hfe_floor_trajs", "hfe_floor_seed.jsonl"),
        help="Path to the suite JSONL (default: seed suite).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=4, help="Simulated runs per prompt.")
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Canonical floor threshold on the family-pass Wilson lower bound.",
    )
    parser.add_argument(
        "--profile", default=None,
        help='Per-capability reliability profile as JSON, e.g. \'{"visual_word_extraction": 0.8}\'.',
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Do not append to the registry log (results are still written).",
    )
    args = parser.parse_args()

    profile = None
    if args.profile:
        profile = dict(DEFAULT_PROFILE)
        profile.update(json.loads(args.profile))

    results_path = os.path.join(ROOT_DIR, "results", "hfe_floor_hootl.txt")
    log_path = None if args.no_log else os.path.join(
        ROOT_DIR, "data", "progress_tracking", "hfe_floor_log.jsonl"
    )
    metrics, report = run_hootl(
        args.suite,
        seed=args.seed,
        runs_per_prompt=args.runs,
        profile=profile,
        threshold=args.threshold,
        results_path=results_path,
        log_path=log_path,
    )
    print(report)
    print(f"report: {results_path}")
    if log_path:
        print(f"log:    {log_path}")


if __name__ == "__main__":
    main()
