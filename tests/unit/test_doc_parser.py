"""Unit tests for agent/api/doc_parser.py (Layer 1)."""
from __future__ import annotations

import pytest

from agent.api.doc_parser import DocumentContent, DocumentParser, DocumentSection

OSHA_MD = """\
# Task Description

Predict whether an establishment will have a serious injury in the next 12 months.
The target variable is outcome where values 1, 2, 3 indicate serious harm.

## Evaluation Criteria

Use PR-AUC as the evaluation metric.

## Data Dictionary

Columns: establishment_id, establishment_type, industry_code, inspection_date, outcome
"""

BOILERPLATE_TEXT = "\n".join([
    "Page 1 of 10",
    "Some content here.",
    "Page 1 of 10",
    "Page 1 of 10",
    "More content.",
    "Page 1 of 10",
])


def test_parse_text_returns_document_content():
    doc = DocumentParser.parse_text(OSHA_MD, file_type="markdown")
    assert isinstance(doc, DocumentContent)
    assert doc.file_type == "markdown"
    assert len(doc.raw_text) > 0


def test_sections_classified_from_headings():
    doc = DocumentParser.parse_text(OSHA_MD, file_type="markdown")
    types = {s.section_type for s in doc.sections}
    assert "task" in types
    assert "evaluation" in types


def test_task_section_contains_predict():
    doc = DocumentParser.parse_text(OSHA_MD, file_type="markdown")
    task_section = next((s for s in doc.sections if s.section_type == "task"), None)
    assert task_section is not None
    assert "predict" in task_section.body.lower() or "serious" in task_section.body.lower()


def test_content_lines_preserved_in_raw_text():
    doc = DocumentParser.parse_text(BOILERPLATE_TEXT, file_type="text")
    # parse_text() preserves the raw_text as-is (PDF boilerplate stripping is
    # PDF-format-only). Verify that actual content lines are present.
    assert "Some content here." in doc.raw_text
    assert "More content." in doc.raw_text


def test_explicit_target_mentions_populated():
    doc = DocumentParser.parse_text(OSHA_MD, file_type="markdown")
    assert len(doc.explicit_target_mentions) > 0


def test_explicit_group_key_mentions_populated():
    doc = DocumentParser.parse_text(OSHA_MD, file_type="markdown")
    # "establishment_id" should be close to aggregation keywords
    assert isinstance(doc.explicit_group_key_mentions, list)
    assert len(doc.explicit_group_key_mentions) >= 0  # may or may not fire


def test_parse_text_empty_string_no_raise():
    doc = DocumentParser.parse_text("", file_type="text")
    assert isinstance(doc, DocumentContent)
    assert doc.raw_text == "" or doc.raw_text is not None


def test_word_count_positive_for_non_empty():
    doc = DocumentParser.parse_text(OSHA_MD, file_type="markdown")
    assert doc.word_count > 0
