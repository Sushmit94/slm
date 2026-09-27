"""Semgrep C/C++ baseline on the processed held-out functions.

Bandit is deliberately not invoked: it analyzes Python and would not be a valid
baseline for Devign's C/C++ functions.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CFG
from src.dataset_utils import normalize_binary_label

def semgrep_label(code: str, rules: str, timeout: int) -> str:
    """A finding is a positive prediction. Failure is raised, never hidden."""
    with tempfile.TemporaryDirectory(prefix="devign_semgrep_") as tmp:
        source = Path(tmp) / "snippet.c"
        source.write_text(code, encoding="utf-8", errors="replace")
        cmd = ["semgrep", "scan", "--config", rules, "--json", "--quiet", "--timeout", str(timeout), str(source)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        # Semgrep returns 0 for clean and commonly 1 for findings.
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"Semgrep failed ({proc.returncode}): {proc.stderr[-500:]}")
        try: findings = json.loads(proc.stdout).get("results", [])
        except json.JSONDecodeError as exc: raise RuntimeError(f"Invalid Semgrep JSON: {proc.stdout[-500:]}") from exc
        return "Vulnerable" if findings else "Not vulnerable"

def main() -> None:
    p = argparse.ArgumentParser(description="Run Semgrep, the applicable C/C++ static baseline.")
    p.add_argument("--test-file", type=Path, default=CFG.processed_dir / "test.jsonl")
    p.add_argument("--output", type=Path, default=CFG.static_predictions_path)
    p.add_argument("--rules", default="p/default", help="Semgrep registry config or local rules YAML.")
    p.add_argument("--timeout", type=int, default=30)
    args = p.parse_args()
    if not shutil.which("semgrep"): raise SystemExit("Semgrep is not installed. Run: pip install -r requirements.txt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.test_file.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            prediction = semgrep_label(row["code"], args.rules, args.timeout)
            dst.write(json.dumps({"id": row["id"], "ground_truth": normalize_binary_label(row["label"]),
                                  "prediction": prediction}) + "\n")
            count += 1
    print(f"Saved {count:,} Semgrep predictions to {args.output}")
    print("Note: Bandit is Python-only, so it is intentionally excluded for C/C++ Devign.")

if __name__ == "__main__": main()
