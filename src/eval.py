"""Evaluate aligned JSONL predictions and produce comparison artifacts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CFG

def load_predictions(path: Path) -> pd.DataFrame:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(rows)
    required = {"id", "ground_truth", "prediction"}
    if not required <= set(df): raise ValueError(f"{path} must contain {required}")
    return df

def score(df: pd.DataFrame, multi_class: bool) -> dict[str, float]:
    average = "weighted" if multi_class else "binary"
    positive = "Vulnerable" if not multi_class else None
    kw = {"average": average, "zero_division": 0}
    if positive: kw["pos_label"] = positive
    return {"precision": precision_score(df.ground_truth, df.prediction, **kw),
            "recall": recall_score(df.ground_truth, df.prediction, **kw),
            "f1": f1_score(df.ground_truth, df.prediction, **kw),
            "accuracy": accuracy_score(df.ground_truth, df.prediction)}

def plot_confusion(df: pd.DataFrame, name: str, out: Path) -> None:
    labels = sorted(set(df.ground_truth) | set(df.prediction))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(df.ground_truth, df.prediction, labels=labels), annot=True, fmt="d",
                cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="Predicted", ylabel="Ground truth", title=f"{name} confusion matrix")
    fig.tight_layout(); fig.savefig(out / f"{name.lower().replace(' ', '_')}_confusion.png", dpi=180); plt.close(fig)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=CFG.model_predictions_path)
    p.add_argument("--static", type=Path, default=CFG.static_predictions_path)
    p.add_argument("--codebert", type=Path, default=CFG.codebert_predictions_path)
    p.add_argument("--multi-class", action="store_true")
    args = p.parse_args(); CFG.metrics_dir.mkdir(parents=True, exist_ok=True); CFG.plots_dir.mkdir(parents=True, exist_ok=True)
    sources = [("Fine-tuned SLM", args.model), ("Static baseline (Semgrep)", args.static)]
    if args.codebert.exists(): sources.append(("CodeBERT", args.codebert))
    results = []
    for name, path in sources:
        if not path.exists(): raise FileNotFoundError(f"Missing {name} predictions: {path}")
        df = load_predictions(path); metrics = score(df, args.multi_class); metrics["approach"] = name; results.append(metrics)
        plot_confusion(df, name, CFG.plots_dir)
        if args.multi_class:
            per_class = pd.DataFrame({"class": sorted(set(df.ground_truth) | set(df.prediction))})
            per_class["f1"] = per_class["class"].map(lambda x: f1_score(df.ground_truth == x, df.prediction == x, zero_division=0))
            per_class.to_csv(CFG.metrics_dir / f"{name.lower().replace(' ', '_')}_per_class_f1.csv", index=False)
    table = pd.DataFrame(results).set_index("approach")[["precision", "recall", "f1", "accuracy"]]
    table.to_csv(CFG.metrics_dir / "comparison.csv", float_format="%.4f")
    (CFG.metrics_dir / "comparison.md").write_text(table.to_markdown(floatfmt=".4f") + "\n", encoding="utf-8")
    ax = table["f1"].plot.bar(color=["#4C78A8", "#F58518", "#54A24B"][:len(table)], ylim=(0, 1), rot=0, ylabel="F1")
    ax.set_title("Held-out vulnerability-detection F1"); ax.figure.tight_layout(); ax.figure.savefig(CFG.plots_dir / "f1_comparison.png", dpi=180); plt.close(ax.figure)
    print(table.to_markdown(floatfmt=".4f")); print(f"Artifacts saved under {CFG.results_dir}")

if __name__ == "__main__": main()
