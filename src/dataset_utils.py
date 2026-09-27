"""Canonical prompt and processed-record utilities.

The marked cell in notebooks/finetune_gemma2_colab.ipynb is intentionally a
verbatim conceptual copy of PROMPT_TEMPLATE and functions below. Keep them in
sync: Colab cannot directly import this local repository module.
"""
from __future__ import annotations
from typing import Any, Mapping

VULNERABLE = "Vulnerable"
NOT_VULNERABLE = "Not vulnerable"
PROMPT_TEMPLATE = """You are a software security reviewer. Analyze the following C/C++ function for a security vulnerability. Reply with exactly one label: Vulnerable or Not vulnerable.\n\nCode:\n```c\n{code}\n```\n\nAnswer:"""

def normalize_binary_label(value: Any) -> str:
    """Map common Devign/HF binary encodings to the experiment's label space."""
    if isinstance(value, bool): return VULNERABLE if value else NOT_VULNERABLE
    if isinstance(value, (int, float)): return VULNERABLE if int(value) == 1 else NOT_VULNERABLE
    text = str(value).strip().lower()
    if text in {"1", "true", "vulnerable", "yes", "positive"}: return VULNERABLE
    return NOT_VULNERABLE

def response_for_record(record: Mapping[str, Any], label_mode: str = "binary") -> str:
    if label_mode == "cwe" and record.get("cwe"):
        return str(record["cwe"])
    return normalize_binary_label(record["label"])

def build_prompt(code: str) -> str:
    return PROMPT_TEMPLATE.format(code=code.strip())

def format_training_text(record: Mapping[str, Any], label_mode: str = "binary") -> str:
    """The exact supervised text: prompt, one space, then target response."""
    return f"{build_prompt(record['code'])} {response_for_record(record, label_mode)}"

def parse_binary_prediction(text: str) -> str:
    """Conservatively parse generated text; check negation before Vulnerable."""
    first_line = text.strip().splitlines()[0].lower() if text.strip() else ""
    if "not vulnerable" in first_line or "not_vulnerable" in first_line:
        return NOT_VULNERABLE
    if "vulnerable" in first_line:
        return VULNERABLE
    return NOT_VULNERABLE  # deterministic fallback, reported by caller if desired
