"""PDF brief extractor.

Reads a PDF and returns plain text plus lightweight cues:
  - mentioned_target:  first column-like word after "target"/"label"
  - task_hint:         one of {classification, regression, clustering, forecasting}
  - data_dict_lines:   best-effort list of "column — description" rows

The heuristics are deliberately simple — good enough to seed a chat
session with context the orchestrator can use. The raw extracted text is
always preserved verbatim under Session.brief.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

_TASK_WORDS = {
    "classification": r"\b(classif\w*|predict\s+the\s+class|binary|multi-?class)\b",
    "regression":     r"\b(regress\w*|predict\s+.*?(value|price|amount|count))\b",
    "clustering":     r"\b(cluster\w*|segment\w*|group\w*\s+customers)\b",
    "forecasting":    r"\b(forecast\w*|time[-\s]?series|seasonal\w*)\b",
}

_TARGET_RE = re.compile(
    # "target column is X", "label: X", "outcome variable = X",
    # "predict X", "dependent variable will be X".
    r"\b(?:target|label|response|outcome|dependent\s+variable)"
    r"(?:\s+(?:column|variable|field|attribute))?"
    r"\s*(?:is|will\s+be|=|:)\s*"
    r"[\"'`]?([A-Za-z_][A-Za-z0-9_]*)[\"'`]?",
    re.IGNORECASE,
)
# Predict-style phrasing: "predict <col>", "we are predicting <col>".
_PREDICT_RE = re.compile(
    r"\b(?:predict(?:ing)?|forecast(?:ing)?)\s+(?:the\s+)?"
    r"[\"'`]?([A-Za-z_][A-Za-z0-9_]*)[\"'`]?",
    re.IGNORECASE,
)
# English stopwords / determiners we never want as a column name.
_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its",
    "we", "you", "our", "their", "his", "her", "is", "are", "was",
    "were", "be", "to", "of", "for", "as", "in", "on", "by", "and",
    "or", "but", "value", "values", "values.", "variable", "column",
    "outcome", "label", "target", "field", "attribute",
}


@dataclass
class BriefSummary:
    text: str
    mentioned_target: str | None
    task_hint: str | None
    data_dict_lines: list[str]


def extract(pdf_bytes: bytes) -> BriefSummary:
    """Return structured cues from a PDF brief.

    Falls back gracefully: a corrupt or image-only PDF yields an empty
    text with no hints, rather than raising.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return BriefSummary(text="[pypdf not installed]",
                            mentioned_target=None, task_hint=None,
                            data_dict_lines=[])

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(pages).strip()
    except Exception as exc:  # noqa: BLE001
        return BriefSummary(text=f"[pdf read failed: {exc}]",
                            mentioned_target=None, task_hint=None,
                            data_dict_lines=[])

    def _pick(regex: re.Pattern[str]) -> str | None:
        for match in regex.finditer(text):
            cand = match.group(1).strip()
            if cand and cand.lower() not in _STOPWORDS and len(cand) > 1:
                return cand
        return None

    target = _pick(_TARGET_RE) or _pick(_PREDICT_RE)

    task_hint = None
    for label, pat in _TASK_WORDS.items():
        if re.search(pat, text, re.IGNORECASE):
            task_hint = label
            break

    # Data-dictionary heuristic: lines that look like "col — description"
    # or "col : description" (various dash styles).
    dict_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or len(line) > 300:
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_ ]{0,40}\s*[-–—:]\s+\S", line):
            dict_lines.append(line)

    return BriefSummary(
        text=text,
        mentioned_target=target,
        task_hint=task_hint,
        data_dict_lines=dict_lines[:200],  # cap to keep responses sane
    )
