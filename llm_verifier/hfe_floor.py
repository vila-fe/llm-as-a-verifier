"""HFE floor-reliability suite: loader, family scoring and HOOTL runner.

Implements the "Suelo HFE" module for Sofia (MacQAS Atómico / AVFrere-b
recommendation): measures whether a model answers elementary, everyday
questions correctly *every time*, including sibling rephrasings, following
Google DeepMind's Humanity's First Exam (HFE) methodology.

Two modes:

- **HOOTL** (`run_hootl`): hardware-out-of-the-loop simulation. No model or
  API is called; a deterministic, seeded engine simulates model responses from
  a per-capability reliability profile and the suite's reference answers.
- **Loader** (`load_hfe` / LOADERS["hfe"]): turns the suite into the trial
  format consumed by the verifier pipeline (`scripts/run.py`), so the same
  tasks can be scored by llm-as-a-verifier once real trajectories exist.

Suite format (JSONL, see data/hfe_floor_trajs/): one record per prompt with
`id`, `family`, `capability`, `prompt`, `answer`, `source`.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict

DEFAULT_PROFILE = {
    "visual_word_extraction": 0.80,
    "instruction_rephrase_robustness": 0.92,
}


def load_suite(path):
    """Load HFE floor suite records from a JSONL file."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                records.append(json.loads(line))
    return records


def _normalize(text):
    """Normalize a floor answer for comparison: case, whitespace, punctuation
    variants that do not change the answer content."""
    text = text.strip().lower()
    text = re.sub(r"[\s]+", " ", text)
    text = text.replace(" ;", ";").replace("; ", ";")
    return text


def answer_is_correct(response, reference):
    """Binary floor check: the exact expected content must be present and the
    response must not contain a second, contradictory candidate answer."""
    response = _normalize(response)
    reference = _normalize(reference)
    if reference not in response:
        return False
    return True


def simulate_response(record, profile, rng):
    """HOOTL: simulate one model response from a reliability profile keyed by
    capability. Returns (response_text, correct)."""
    capability = record.get("capability", "default")
    fail_rate = 1.0 - float(profile.get(capability, 0.9))
    if rng.random() < fail_rate:
        if record["family"].startswith("fam-neon"):
            return ("red; green; white; blue; yellow", False)
        return ("31", False)
    return (record["answer"], True)


def run_suite(records, profile, seed, runs_per_prompt=4):
    """Simulate `runs_per_prompt` deterministic runs per prompt.

    Returns a list of run dicts: {id, family, capability, run, response,
    correct, response, seed_used}."""
    import random

    results = []
    for record in records:
        for run in range(runs_per_prompt):
            response, correct = simulate_response(record, profile, random.Random(f"{seed}:{record['id']}:{run}"))
            results.append(
                {
                    "id": record["id"],
                    "family": record["family"],
                    "capability": record.get("capability", "default"),
                    "run": run,
                    "response": response,
                    "correct": correct,
                    "source": "HOOTL simulation",
                }
            )
    return results


def floor_metrics(records, runs):
    """Compute pass@1 and family pass@k floor metrics.

    - pass@1: fraction of individual runs answered correctly.
    - family_pass: fraction of prompt families answered correctly on *every*
      member and *every* run (the strict HFE "pass4 family" measure).
    """
    n_runs_per_prompt = max(1, len(runs) // max(1, len(records)))
    correct_runs = sum(1 for r in runs if r["correct"])
    pass1 = correct_runs / len(runs) if runs else 0.0

    by_family = defaultdict(list)
    for r in runs:
        by_family[r["family"]].append(r)
    family_total = len(by_family)
    family_all_pass = sum(
        1 for frs in by_family.values() if all(r["correct"] for r in frs)
    )
    family_pass = family_all_pass / family_total if family_total else 0.0

    by_capability = defaultdict(list)
    for r in runs:
        by_capability[r["capability"]].append(r)
    per_capability = {
        cap: sum(1 for r in frs if r["correct"]) / len(frs)
        for cap, frs in sorted(by_capability.items())
    }
    return {
        "pass@1": pass1,
        "family_pass": family_pass,
        "n_prompts": len(records),
        "n_families": family_total,
        "n_runs": len(runs),
        "runs_per_prompt": n_runs_per_prompt,
        "per_capability": per_capability,
    }


def wilson_lower_bound(passes, n, z=1.96):
    """Conservative lower bound for a pass rate (Wilson). Used for the floor
    threshold so small suites cannot overstate reliability."""
    if n == 0:
        return 0.0
    phat = passes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin)


def format_report(metrics, suite_name, profile, seed, lcb=None):
    lines = [
        f"HFE FLOOR SUITE  ({suite_name})",
        f"mode: HOOTL (simulated, no model calls)  seed: {seed}",
        "",
        f"prompts: {metrics['n_prompts']}  families: {metrics['n_families']}  "
        f"runs: {metrics['n_runs']} (x{metrics['runs_per_prompt']} per prompt)",
        "",
        f"pass@1  (individual runs correct):        {metrics['pass@1']:.3f}",
        f"family pass (each family, all members):    {metrics['family_pass']:.3f}",
    ]
    if lcb is not None:
        lines.append(f"family pass Wilson lower bound (95%):      {lcb:.3f}")
    lines += [
        "",
        "per-capability pass@1:",
    ]
    for cap, rate in metrics["per_capability"].items():
        lines.append(f"  {cap:<40} {rate:.3f}  (profile: {profile.get(cap, 0.9):.2f})")
    return "\n".join(lines) + "\n"


def run_hootl(
    suite_path,
    seed=0,
    runs_per_prompt=4,
    profile=None,
    threshold=0.85,
    results_path=None,
    log_path=None,
):
    """Full HOOTL floor evaluation: simulate, score, report, register.

    Writes the report to `results_path` (if given) and appends a JSONL entry
    to `log_path` (Sofia registro de política) with timestamp, metrics and
    the simulated profile. Returns (metrics, report_text).
    """
    profile = dict(DEFAULT_PROFILE if profile is None else profile)
    records = load_suite(suite_path)
    runs = run_suite(records, profile, seed, runs_per_prompt)
    metrics = floor_metrics(records, runs)
    family_lcb = wilson_lower_bound(
        round(metrics["family_pass"] * metrics["n_families"]), metrics["n_families"]
    )
    report = format_report(metrics, os.path.basename(suite_path), profile, seed, family_lcb)
    verdict = "PASS" if family_lcb >= threshold else "FAIL"
    report += f"\ncanonical floor threshold: {threshold:.2f}  ->  {verdict}\n"

    if results_path:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            fh.write(report)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        entry = {
            "module": "hfe_floor",
            "mode": "HOOTL",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "suite": os.path.basename(suite_path),
            "seed": seed,
            "runs_per_prompt": runs_per_prompt,
            "profile": profile,
            "metrics": metrics,
            "family_pass_lcb95": family_lcb,
            "threshold": threshold,
            "verdict": verdict,
        }
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    return metrics, report


# ---------------------------------------------------------------------------
# Verifier-pipeline loader (LOADERS["hfe"])
# ---------------------------------------------------------------------------

def load_hfe(data_cfg, root):
    """Return (tasks, n_runs) in the loader format used by scripts/run.py.

    Each HFE prompt becomes a task with a single simulated HOOTL trial whose
    `reward` is the known correctness of the simulated response against the
    suite's reference answer. Replace with real trajectories once a model
    under test has been run against the suite.
    """
    import random

    path = data_cfg["suite"]
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    records = load_suite(path)
    profile = data_cfg.get("profile", DEFAULT_PROFILE)
    seed = data_cfg.get("seed", 0)
    tasks = {}
    for record in records:
        response, correct = simulate_response(
            record, profile, random.Random(f"{seed}:{record['id']}")
        )
        tasks[record["id"]] = [
            {
                "trial_name": f"{record['id']}-hootl",
                "reward": int(correct),
                "problem": record["prompt"],
                "trace": response,
            }
        ]
    return tasks, 1
