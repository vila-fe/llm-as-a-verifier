#!/usr/bin/env python3
"""Ingest the Kaggle `deepmind/hfe-public` dataset (Humanity's First Exam,
112 public prompts, CC BY 4.0) into the Suelo HFE suite JSONL format.

Handles any of the following inputs (auto-detected):
  - path to the zip downloaded from Kaggle                 (.zip)
  - the extracted dataset directory                        (dir)
  - a single JSON / JSONL / CSV / YAML file with the prompts

It recognizes common field spellings (`task_id`/`id`, `question`/`prompt`,
`reference_answer`/`answer`/`solution`, `family_id`/`family`, `category`/
`capability`, Harbor task folders with `task.yaml`/`task.json`/`prompt.*`,
and nested `records`/`data`/`tasks` wrappers). Output is one JSONL record
per prompt in the Suelo HFE schema:

    {"id", "family", "capability", "prompt", "answer", "source"}

After ingestion it optionally runs the HOOTL floor evaluation on the new
suite (nothing is simulated about the *dataset* — prompts and reference
answers are the real HFE data; only the model behavior is simulated).

Usage:
    python scripts/ingest_hfe_public.py ZIP_O_FILE [--out data/hfe_floor_trajs/hfe_public_floor.jsonl]
    python scripts/ingest_hfe_public.py ZIP_O_FILE --run              # ingest + HOOTL
"""

import argparse
import csv
import json
import os
import re
import sys
import zipfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

DEFAULT_OUT = os.path.join(ROOT_DIR, "data", "hfe_floor_trajs", "hfe_public_floor.jsonl")

ID_KEYS = ["task_id", "id", "prompt_id", "task_name", "name"]
FAMILY_KEYS = ["family_id", "family", "family_name", "task_family", "group_id"]
PROMPT_KEYS = ["question", "prompt", "input", "instruction", "text", "problem"]
ANSWER_KEYS = ["reference_answer", "reference", "answer", "solution", "expected",
               "expected_answer", "gold", "target", "ideal"]
CATEGORY_KEYS = ["category", "capability", "sub_category", "subcategory",
                 "capability_category", "topic", "macro_category"]
SOURCE_TAG = "HFE public (Kaggle deepmind/hfe-public, CC BY 4.0)"


def _first(d, keys):
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return v
    return None


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_") or "unknown"


def _unwrap(payload):
    """Unwrap common container keys inside a loaded JSON document."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("records", "data", "items", "tasks", "prompts"):
            if isinstance(payload.get(k), list):
                return payload[k]
    return [payload] if isinstance(payload, dict) else []


def _capability(value):
    """Normalize an HFE capability/category label into a suite capability key."""
    return _slug(value) if value else "unspecified"


def _record_from(d, idx):
    prompt = _first(d, PROMPT_KEYS)
    answer = _first(d, ANSWER_KEYS)
    if prompt is None:
        return None
    rid = _first(d, ID_KEYS) or f"hfe-pub-{idx:03d}"
    family = _first(d, FAMILY_KEYS) or f"fam-{rid}"
    cap = _capability(_first(d, CATEGORY_KEYS))
    return {
        "id": str(rid),
        "family": f"fam-{_slug(family)}",
        "capability": cap,
        "prompt": str(prompt),
        "answer": str(answer) if answer is not None else "",
        "source": SOURCE_TAG,
    }


def ingest_rows(rows):
    records, seen = [], set()
    for idx, row in enumerate(rows):
        rec = _record_from(row, idx)
        if rec and rec["id"] not in seen:
            seen.add(rec["id"])
            records.append(rec)
    return records


def from_json_obj(payload):
    return ingest_rows(_unwrap(payload))


def from_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.extend(_unwrap(json.loads(line)))
                except json.JSONDecodeError:
                    continue
    return ingest_rows(rows)


def from_csv(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: v for k, v in row.items() if k})
    return ingest_rows(rows)


def _iter_prompt_files(dir_path):
    """Recursive walk over the dataset tree, yielinging loose prompt files."""
    for base, _dirs, files in os.walk(dir_path):
        for e in sorted(files):
            low = e.lower()
            if low.endswith((".jsonl", ".json", ".csv")):
                yield os.path.join(base, e)


def _iter_harbor_dirs(dir_path):
    """Yield Harbor-style leaf task dirs (containing task.yaml / task.json /
    prompt.*, directly)."""
    for base, dirs, _files in os.walk(dir_path):
        for d in sorted(dirs):
            p = os.path.join(base, d)
            markers = [f for f in os.listdir(p)
                       if f.lower() in ("task.yaml", "task.yml", "task.json", "config.yaml", "config.json")
                       or f.lower().startswith("prompt")]
            if markers:
                yield p


def from_dir(dir_path):
    """Scan an extracted dataset directory recursively. Handles loose prompt
    files (JSON/JSONL/CSV) and Harbor-style per-task folders (task.yaml /
    task.json / prompt.txt)."""
    all_records = []
    for p in _iter_prompt_files(dir_path):
        low = p.lower()
        try:
            if low.endswith(".jsonl"):
                all_records += from_jsonl(p)
            elif low.endswith(".json"):
                all_records += from_json_obj(json.load(open(p, encoding="utf-8")))
            elif low.endswith(".csv"):
                all_records += from_csv(p)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
    for d in _iter_harbor_dirs(dir_path):
        tdir = {"task_name": os.path.basename(d)}
        for cand in ("task.yaml", "task.yml", "task.json", "config.yaml", "config.json"):
            p = os.path.join(d, cand)
            if not os.path.exists(p):
                continue
            if cand.endswith("json"):
                try:
                    tdir.update(json.load(open(p, encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    continue
            else:
                try:
                    import yaml
                except ImportError:
                    continue
                try:
                    tdir.update(yaml.safe_load(open(p, encoding="utf-8")) or {})
                except Exception:
                    continue
            break
        if not _first(tdir, PROMPT_KEYS):
            for f in sorted(os.listdir(d)):
                if f.lower().startswith("prompt") and f.lower().endswith((".txt", ".md")):
                    tdir["prompt"] = open(os.path.join(d, f), encoding="utf-8").read()
                    break
        if not _first(tdir, PROMPT_KEYS):
            continue
        rec = _record_from(
            {
                **tdir,
                "id": tdir.get("id") or tdir.get("task_name"),
                "family": tdir.get("family") or tdir.get("task_name"),
            },
            len(all_records),
        )
        if rec:
            all_records.append(rec)
    return _dedupe(all_records)


def _dedupe(records):
    seen, out = set(), []
    for r in records:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def from_zip(zip_path):
    """Ingest from the Kaggle zip: extract to memory and delegate to the
    same auto-detection used for directories/files."""
    zf = zipfile.ZipFile(zip_path)
    names = [n for n in zf.namelist() if not n.endswith("/")]
    all_records = []
    # Try whole-archive directory emulation
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        zf.extractall(td)
        # find the deepest common root directory
        root = td
        while True:
            sub = [e for e in os.listdir(root) if not e.startswith(".")]
            if len(sub) == 1 and os.path.isdir(os.path.join(root, sub[0])):
                root = os.path.join(root, sub[0])
                continue
            break
        all_records = from_dir(root)
    return all_records


def ingest(path):
    low = str(path).lower()
    if low.endswith(".zip"):
        return from_zip(path)
    if os.path.isdir(path):
        return from_dir(path)
    if low.endswith(".jsonl"):
        return from_jsonl(path)
    if low.endswith(".json"):
        return from_json_obj(json.load(open(path, encoding="utf-8")))
    if low.endswith(".csv"):
        return from_csv(path)
    raise SystemExit(f"Formato no reconocido para '{path}'. Usa .zip, dir, .json, .jsonl o .csv.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Zip, directorio o fichero del dataset hfe-public.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--run", action="store_true",
                        help="Ejecutar la evaluacion HOOTL sobre la suite ingerida tras la ingesta.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=4)
    args = parser.parse_args()

    records = ingest(args.input)
    if not records:
        raise SystemExit(
            f"No se detectaron prompts con respuesta en '{args.input}'.\n"
            "Revisa el contenido del zip/directorio: el script esperaba ficheros "
            ".json/.jsonl/.csv de prompts o carpetas de tareas Harbor (task.yaml / task.json / prompt.txt)."
        )
    n_answered = sum(1 for r in records if r["answer"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    n_families = len({r["family"] for r in records})
    print(f"Ingestados {len(records)} prompts "
          f"({n_families} familias, {n_answered} con respuesta de referencia) -> {args.out}")
    if not args.run:
        print("Validacion HOOTL de la suite ingerida:")
    if args.run:
        from llm_verifier.hfe_floor import run_hootl

        results_path = os.path.join(ROOT_DIR, "results", "hfe_floor_hootl.txt")
        log_path = os.path.join(ROOT_DIR, "data", "progress_tracking", "hfe_floor_log.jsonl")
        metrics, report = run_hootl(
            args.out,
            seed=args.seed,
            runs_per_prompt=args.runs,
            results_path=results_path,
            log_path=log_path,
        )
        print(report)
        print(f"report: {results_path}")
        print(f"log:    {log_path}")
    else:
        print(f"    python scripts/run_hfe_floor.py --suite {args.out}")


if __name__ == "__main__":
    main()
