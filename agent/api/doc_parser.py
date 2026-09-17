"""Layer 1 — Multi-format document parser (no LLM calls).

Parses PDF / MD / TXT / DOCX / HTML files into a clean DocumentContent
dataclass with section detection and explicit keyword extraction.

Per INSTRUCTION_UNDERSTANDING_FIX.md (Section 3 + Refinement v1.1):
  - PDF: pdfplumber line extraction with dual boilerplate detection
    (repeated-line normalised + geometric/font-size anomaly).
    pymupdf fallback for scanned PDFs.
  - Markdown / Text: read straight, preserve heading structure.
  - DOCX / HTML: use python-docx / BeautifulSoup if installed.

Returns a deterministic DocumentContent with hint lists populated from the
text so the downstream extractor and enricher don't have to re-grep.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentSection:
    heading: str
    body: str
    section_type: str = "other"   # task | data_description | evaluation | instructions | other


@dataclass
class DocumentContent:
    raw_text: str
    sections: list[DocumentSection]
    file_type: str
    page_count: int
    word_count: int
    has_structured_headings: bool
    explicit_target_mentions: list[str] = field(default_factory=list)
    explicit_metric_mentions: list[str] = field(default_factory=list)
    explicit_task_mentions: list[str] = field(default_factory=list)
    explicit_group_key_mentions: list[str] = field(default_factory=list)


class DocumentParser:
    """Deterministic parser. No LLM calls."""

    TASK_KEYWORDS = [
        "predict", "classify", "forecast", "detect", "identify",
        "estimate", "score", "rank", "cluster", "segment",
        "classification", "regression", "anomaly", "survival",
    ]
    METRIC_KEYWORDS = [
        "auc", "roc", "pr-auc", "precision", "recall", "f1",
        "accuracy", "rmse", "mae", "r2", "r-squared", "log loss",
        "average precision", "pr_auc", "roc_auc",
    ]
    TARGET_KEYWORDS = [
        "target", "outcome", "label", "predict", "dependent",
        "response", "y variable", "output variable",
    ]
    GROUP_KEYWORDS = [
        "aggregate", "group by", "per establishment", "per entity",
        "entity level", "roll up", "summarise", "summarize",
        "each establishment", "each customer", "each patient",
    ]
    SECTION_HEADING_PATTERNS = [
        r"^#{1,4}\s+(.+)$",
        r"^([A-Z][A-Z\s]{3,30}):?\s*$",
        r"^\*\*(.+)\*\*\s*$",
        r"^(\d+\.?\s+[A-Z].{5,50})$",
    ]

    HEADING_TYPE_MAP = {
        "task": ["task", "objective", "goal", "problem", "challenge",
                 "purpose", "aim"],
        "data_description": ["data", "dataset", "column", "field",
                             "variable", "feature", "attribute",
                             "dictionary", "schema"],
        "evaluation": ["evaluation", "metric", "measure", "performance",
                       "scoring", "criteria", "kpi"],
        "instructions": ["instruction", "requirement", "deliverable",
                         "output", "expected", "you must", "please"],
    }

    # ── Public entry ─────────────────────────────────────────────────────
    def parse(self, file_path: str | Path) -> DocumentContent:
        path = Path(file_path)
        suffix = path.suffix.lower()

        try:
            if suffix == ".pdf":
                raw = self._parse_pdf(path)
            elif suffix in (".md", ".markdown"):
                raw = self._parse_markdown(path)
            elif suffix in (".txt", ".text"):
                raw = self._parse_text(path)
            elif suffix == ".docx":
                raw = self._parse_docx(path)
            elif suffix in (".html", ".htm"):
                raw = self._parse_html(path)
            else:
                raw = path.read_text(errors="replace")
        except Exception as exc:  # noqa: BLE001 — never raise to caller
            raw = f"[doc_parser failed: {exc}]"

        sections = self._split_sections(raw)
        self._classify_sections(sections)

        return DocumentContent(
            raw_text=raw,
            sections=sections,
            file_type=suffix.lstrip(".") or "unknown",
            page_count=self._count_pages(path, suffix),
            word_count=len(raw.split()),
            has_structured_headings=any(s.heading for s in sections),
            explicit_target_mentions=self._extract_near_keyword(raw, self.TARGET_KEYWORDS),
            explicit_metric_mentions=self._extract_near_keyword(raw, self.METRIC_KEYWORDS),
            explicit_task_mentions=self._extract_near_keyword(raw, self.TASK_KEYWORDS),
            explicit_group_key_mentions=self._extract_near_keyword(raw, self.GROUP_KEYWORDS),
        )

    @classmethod
    def parse_text(cls, text: str, file_type: str = "text") -> DocumentContent:
        """Parse a raw text string (used when brief is in-memory rather than on disk)."""
        parser = cls()
        sections = parser._split_sections(text or "")
        parser._classify_sections(sections)
        return DocumentContent(
            raw_text=text or "",
            sections=sections,
            file_type=file_type,
            page_count=0,
            word_count=len((text or "").split()),
            has_structured_headings=any(s.heading for s in sections),
            explicit_target_mentions=parser._extract_near_keyword(text or "", parser.TARGET_KEYWORDS),
            explicit_metric_mentions=parser._extract_near_keyword(text or "", parser.METRIC_KEYWORDS),
            explicit_task_mentions=parser._extract_near_keyword(text or "", parser.TASK_KEYWORDS),
            explicit_group_key_mentions=parser._extract_near_keyword(text or "", parser.GROUP_KEYWORDS),
        )

    # ── Format-specific parsers ──────────────────────────────────────────
    def _parse_pdf(self, path: Path) -> str:
        """pdfplumber + dual boilerplate detection, pymupdf fallback."""
        try:
            import pdfplumber
        except ImportError:
            return self._parse_pdf_pymupdf(path)

        import statistics
        from collections import defaultdict

        pages: list[str] = []
        page_boilerplate_lines: set[str] = set()

        try:
            with pdfplumber.open(path) as pdf:
                # Strategy B: geometric / font-size detection
                all_font_sizes: list[float] = []
                for page in pdf.pages:
                    try:
                        for ch in page.chars:
                            if "size" in ch:
                                all_font_sizes.append(ch["size"])
                    except Exception:
                        pass
                body_size = (statistics.median(all_font_sizes)
                             if all_font_sizes else 10.0)

                for page in pdf.pages:
                    page_height = page.height or 1.0
                    top_band = page_height * 0.08
                    bottom_band = page_height * 0.92
                    geometric_boilerplate: set[str] = set()
                    try:
                        words = page.extract_words(
                            extra_attrs=["size"], use_text_flow=True
                        )
                        by_line: dict[float, list[dict]] = defaultdict(list)
                        for w in words:
                            y_key = round(w["top"] / 2) * 2
                            by_line[y_key].append(w)
                        for y, line_words in by_line.items():
                            if not line_words:
                                continue
                            line_text = " ".join(w["text"] for w in line_words).strip()
                            if not line_text:
                                continue
                            sizes = [w.get("size", body_size) for w in line_words
                                     if w.get("size") is not None]
                            line_size = statistics.median(sizes) if sizes else body_size
                            size_ratio = abs(line_size - body_size) / max(body_size, 1e-6)
                            in_margin = (y < top_band or y > bottom_band)
                            size_anomaly = size_ratio > 0.20
                            if in_margin and size_anomaly:
                                geometric_boilerplate.add(line_text)
                    except Exception:
                        pass
                    page_boilerplate_lines.update(geometric_boilerplate)
                    text = page.extract_text() or ""
                    pages.append(text)
        except Exception:
            return self._parse_pdf_pymupdf(path)

        # Strategy A: repeated-line detection (normalise digits to 'N')
        if len(pages) >= 3:
            line_counts: dict[str, int] = {}
            for page_text in pages:
                for line in page_text.splitlines():
                    stripped = line.strip()
                    if stripped:
                        normalised = re.sub(r"\b\d+\b", "N", stripped)
                        line_counts[normalised] = line_counts.get(normalised, 0) + 1
            page_count = len(pages)
            threshold = max(3, page_count // 2)
            repeated_boilerplate_norm = {
                norm for norm, count in line_counts.items() if count >= threshold
            }
        else:
            repeated_boilerplate_norm = set()

        cleaned: list[str] = []
        for page_text in pages:
            keep_lines = []
            for line in page_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    keep_lines.append(line)
                    continue
                if stripped in page_boilerplate_lines:
                    continue
                normalised = re.sub(r"\b\d+\b", "N", stripped)
                if normalised in repeated_boilerplate_norm:
                    continue
                keep_lines.append(line)
            cleaned.append("\n".join(keep_lines))

        result = "\n\n--- PAGE BREAK ---\n\n".join(cleaned)

        # pymupdf fallback for scanned/image PDFs
        if len(result.strip()) < 100:
            fallback = self._parse_pdf_pymupdf(path)
            if len(fallback.strip()) > len(result.strip()):
                return fallback
        return result

    @staticmethod
    def _parse_pdf_pymupdf(path: Path) -> str:
        try:
            import fitz  # pymupdf
            doc = fitz.open(path)
            text = "\n\n--- PAGE BREAK ---\n\n".join(
                page.get_text() for page in doc
            )
            doc.close()
            return text
        except Exception:
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                return "\n\n--- PAGE BREAK ---\n\n".join(
                    (p.extract_text() or "") for p in reader.pages
                )
            except Exception:
                return ""

    def _parse_markdown(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    def _parse_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    def _parse_docx(self, path: Path) -> str:
        try:
            from docx import Document
        except ImportError:
            return path.read_text(errors="replace")
        doc = Document(str(path))
        parts: list[str] = []
        for para in doc.paragraphs:
            if para.style and para.style.name and para.style.name.startswith("Heading"):
                parts.append(f"## {para.text}")
            else:
                parts.append(para.text)
        return "\n".join(parts)

    def _parse_html(self, path: Path) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return path.read_text(errors="replace")
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    # ── Section splitting ────────────────────────────────────────────────
    def _split_sections(self, text: str) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        current_heading = ""
        current_lines: list[str] = []
        for line in text.splitlines():
            heading_match = None
            stripped = line.strip()
            for pattern in self.SECTION_HEADING_PATTERNS:
                m = re.match(pattern, stripped)
                if m:
                    heading_match = m.group(1).strip()
                    break
            if heading_match:
                if current_lines or current_heading:
                    sections.append(DocumentSection(
                        heading=current_heading,
                        body="\n".join(current_lines).strip(),
                    ))
                current_heading = heading_match
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines or current_heading:
            sections.append(DocumentSection(
                heading=current_heading,
                body="\n".join(current_lines).strip(),
            ))
        if not sections:
            sections.append(DocumentSection(heading="", body=text))
        return sections

    def _classify_sections(self, sections: list[DocumentSection]) -> None:
        for section in sections:
            h = section.heading.lower()
            if not h:
                continue
            for stype, kws in self.HEADING_TYPE_MAP.items():
                if any(kw in h for kw in kws):
                    section.section_type = stype
                    break

    # ── Keyword extraction ───────────────────────────────────────────────
    _STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "each", "this",
        "that", "these", "those", "and", "or", "but", "if", "then", "because",
    })

    def _extract_near_keyword(self, text: str, keywords: list[str]) -> list[str]:
        if not text:
            return []
        tokens = text.lower().split()
        found: list[str] = []
        kw_set = set(keywords)
        for i, token in enumerate(tokens):
            if any(kw in token for kw in kw_set):
                context = tokens[max(0, i - 5):i + 10]
                found.extend(context)
        # Strip punctuation, drop stopwords, dedupe preserving order
        cleaned: list[str] = []
        seen: set[str] = set()
        for t in found:
            t2 = re.sub(r"[^a-z0-9_]", "", t)
            if not t2 or t2 in self._STOPWORDS or t2 in seen:
                continue
            seen.add(t2)
            cleaned.append(t2)
        return cleaned

    def _count_pages(self, path: Path, suffix: str) -> int:
        if suffix != ".pdf":
            return 0
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return len(pdf.pages)
        except Exception:
            try:
                from pypdf import PdfReader
                return len(PdfReader(str(path)).pages)
            except Exception:
                return 0
