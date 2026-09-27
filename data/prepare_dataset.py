"""Prepare a leakage-aware Devign-style dataset as portable JSONL splits."""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CFG
from src.dataset_utils import build_prompt, response_for_record

def norm_code(code: str) -> str:
    """Cheap near-duplicate key: remove comments, whitespace, and case variation.

    This intentionally catches formatting/comment variants. It is not semantic
    clone detection; report this limitation in coursework and upgrade it later
    with a token/minhash clone detector if needed.
    """
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", str(code), flags=re.S)
    return re.sub(r"\s+", "", code).lower()

def canonicalize(raw: Any, label_mode: str) -> list[dict[str, Any]]:
    """Small dataset-specific adapter. Extend this function for BigVul/DiverseVul."""
    rows = []
    for i, ex in enumerate(raw):
        code = next((ex[k] for k in ("func", "function", "code", "source") if k in ex and ex[k]), None)
        label = next((ex[k] for k in ("target", "label", "vul", "vulnerability") if k in ex), None)
        if code is None or label is None: continue
        cwe = next((ex[k] for k in ("cwe", "cwe_id", "CWE") if k in ex and ex[k]), None)
        row = {"id": str(ex.get("id", ex.get("commit_id", i))), "code": str(code), "label": label}
        if cwe: row["cwe"] = str(cwe)
        row["response"] = response_for_record(row, label_mode)
        row["prompt"] = build_prompt(row["code"])
        rows.append(row)
    return rows

def load_source(dataset_name: str, dataset_path: str | None) -> DatasetDict:
    if dataset_path:
        return load_dataset("json", data_files=dataset_path)
    # Default is DetectVul/devign; alternate Hugging Face datasets need only a
    # config change plus any field mapping needed in canonicalize().
    hf_name = "DetectVul/devign" if dataset_name.lower() == "devign" else dataset_name
    return load_dataset(hf_name)

def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, unique = set(), []
    for row in rows:
        key = hashlib.sha256(norm_code(row["code"]).encode()).hexdigest()
        if key not in seen:
            seen.add(key); unique.append(row)
    print(f"Deduplication: retained {len(unique):,}/{len(rows):,}; removed {len(rows)-len(unique):,} exact/normalized near duplicates.")
    return unique

def split_rows(rows: list[dict[str, Any]], cfg=CFG) -> dict[str, list[dict[str, Any]]]:
    # We split after global deduplication so no normalized duplicate can cross splits.
    labels = [r["response"] for r in rows]
    train, remaining = train_test_split(rows, test_size=cfg.val_ratio + cfg.test_ratio,
        random_state=cfg.random_seed, stratify=labels)
    remaining_labels = [r["response"] for r in remaining]
    val_share = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    val, test = train_test_split(remaining, test_size=1 - val_share,
        random_state=cfg.random_seed, stratify=remaining_labels)
    return {"train": train, "validation": val, "test": test}

def write_splits(splits: dict[str, list[dict[str, Any]]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows: fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name:10} {len(rows):,} rows | {dict(Counter(r['response'] for r in rows))}")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=CFG.dataset_name)
    p.add_argument("--dataset-path", default=CFG.dataset_path)
    p.add_argument("--label-mode", choices=["binary", "cwe"], default=CFG.label_mode)
    p.add_argument("--output-dir", type=Path, default=CFG.processed_dir)
    args = p.parse_args()
    source = load_source(args.dataset, args.dataset_path)
    # Remove duplicates globally, including across official partitions. If an
    # official train/validation/test partition exists, retain its membership;
    # otherwise make a reproducible stratified partition.
    rows = []
    for split_name, split in source.items():
        converted = canonicalize(split, args.label_mode)
        for row in converted:
            row["_source_split"] = split_name
        rows.extend(converted)
    rows = deduplicate(rows)
    print("\nSplit statistics:")
    official = {"train", "validation", "test"}
    if official <= set(source.keys()):
        splits = {name: [r for r in rows if r["_source_split"] == name] for name in official}
        print("Using the dataset's official split after global deduplication.")
    else:
        print("No complete official split found; using stratified random split.")
        splits = split_rows(rows)
    for partition in splits.values():
        for row in partition: row.pop("_source_split", None)
    write_splits(splits, args.output_dir)

if __name__ == "__main__": main()
