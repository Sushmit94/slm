# SLM vulnerability detection: Gemma 2 vs. static analysis

This course-project scaffold evaluates C/C++ function-level vulnerability detection on one held-out test set. It compares a QLoRA-fine-tuned `google/gemma-2-2b-it` adapter with Semgrep; a CodeBERT prediction file can be added later without changing the evaluator.

The default corpus is Devign. Dataset handling is centralized in `data/prepare_dataset.py`: adapting BigVul or DiverseVul means changing `dataset_name` in `src/config.py` and extending the small `canonicalize()` adapter, rather than replacing the pipeline.

## Layout

- `data/prepare_dataset.py` downloads/normalizes, globally deduplicates, preserves an official split when available (otherwise stratifies), and writes JSONL.
- `src/dataset_utils.py` owns the training/inference prompt contract.
- `notebooks/finetune_gemma2_colab.ipynb` trains only in Colab; local scripts do not require a GPU.
- `src/baseline_static.py` emits Semgrep predictions in the model-compatible schema.
- `src/eval.py` writes tables and plots.

## Local setup and data preparation

Use Python 3.10+ in a virtual environment:

```powershell
cd slm-vuln-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data/prepare_dataset.py
```

The script first loads `DetectVul/devign` through Hugging Face `datasets`; use a local JSON/JSONL source instead with `--dataset-path path/to/file.jsonl`. It saves `train.jsonl`, `validation.jsonl`, and `test.jsonl` in `data/processed/`. Each record has `id`, `code`, `label`, `prompt`, and `response`; models and evaluators use this common representation.

Before splitting, the script removes functions with the same normalized code (comments, whitespace, and case removed) globally. This prevents those common Devign formatting duplicates from leaking across splits. It is deliberately documented as a lightweight *near-duplicate* defense, not a substitute for full semantic clone detection.

## Colab fine-tuning

1. In Google Drive, create `MyDrive/slm-vuln-detection/data/processed/` and upload all three processed JSONL files there.
2. Open `notebooks/finetune_gemma2_colab.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU**.
3. Accept Gemma's Hugging Face access terms and authenticate when required.
4. Run cells top-to-bottom. The adapter is saved to `MyDrive/slm-vuln-detection/models/gemma2-devign-lora/`; test predictions go to `MyDrive/slm-vuln-detection/results/model_predictions.jsonl`.

The notebook has an explicitly marked copy of the prompt functions from `src/dataset_utils.py`. Colab cannot import this local module, so keep those two locations synchronized. The prompt and prediction parser are identical in concept by design.

## Static baseline and final evaluation

Copy the Colab file `model_predictions.jsonl` to local `results/model_predictions.jsonl`, then run:

```powershell
python src/baseline_static.py --test-file data/processed/test.jsonl
python src/eval.py
```

Semgrep is the appropriate static-analysis baseline for Devign's C/C++ functions. Bandit is intentionally excluded because it analyzes Python, not C/C++. Semgrep's `p/default` configuration is the default; for a course report, record the exact `--rules` configuration/version you use, e.g. `python src/baseline_static.py --rules p/default`.

Evaluation saves `results/metrics/comparison.csv`, `results/metrics/comparison.md`, per-method confusion-matrix PNGs, and an F1 comparison chart. To include CodeBERT later, write `results/codebert_predictions.jsonl` with one JSON object per line:

```json
{"id":"example-id", "ground_truth":"Vulnerable", "prediction":"Not vulnerable"}
```

Then run `python src/eval.py` again. For CWE/multiclass work, prepare records with `--label-mode cwe` only if the chosen source provides reliable CWE labels, and invoke `python src/eval.py --multi-class`.
