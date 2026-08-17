from pathlib import Path
import json
import re

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


# ============================================================================
# DIRECTORIES
# ============================================================================

RAW_DIR = Path("data/raw/3gpp")
PROCESSED_DIR = Path("data/processed/3gpp")


# ============================================================================
# TEXT UTILITIES
# ============================================================================


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving meaningful technical content."""

    if not text:
        return ""

    text = str(text)

    text = text.replace("\xa0", " ")
    text = text.replace("\t", " ")

    # Convert line breaks inside headings/content into spaces where
    # appropriate, while still retaining paragraph boundaries externally.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)

    return text.strip()


# ============================================================================
# SECTION DETECTION
# ============================================================================


def extract_section_info(text: str):
    """
    Extract an explicit 3GPP section number and title.

    Supported examples:

        1 Scope
        4.2 Architecture reference model
        4.2.2 Network Functions and entities
        4.2.5a Radio Capabilities Signalling optimisation
        Annex F (informative): Redundant user plane paths

    The Annex title may contain line breaks in the source DOCX.
    """

    text = clean_text(text)

    pattern = (
        r"^("
        r"Annex\s+[A-Za-z]+"
        r"|"
        r"\d+(?:\.\d+)*[A-Za-z]?"
        r")"
        r"(?:\s*:\s*|\s+)"
        r"([\s\S]+)$"
    )

    match = re.match(pattern, text)

    if not match:
        return None, text

    section_number = match.group(1).strip()
    title = clean_text(match.group(2))

    return section_number, title


def is_heading(paragraph: Paragraph) -> bool:
    """Return True when the DOCX paragraph uses a Heading style."""

    style = paragraph.style.name or ""

    return style.startswith("Heading ")


def get_heading_level(paragraph: Paragraph) -> int:
    """Convert Heading 1 -> 1, Heading 2 -> 2, etc."""

    style = paragraph.style.name or ""

    match = re.match(r"Heading\s+(\d+)", style)

    if match:
        return int(match.group(1))

    return 0


def detect_section_type(section_number, title=""):
    """
    Classify section metadata.

    Values:
        cover
        numbered
        annex
        unnumbered
    """

    if section_number and str(section_number).lower().startswith("annex "):
        return "annex"

    if section_number:
        return "numbered"

    return "unnumbered"


# ============================================================================
# SECTION HIERARCHY
# ============================================================================


def build_parent_section(section_stack, level):
    """Find the nearest numbered parent section."""

    for previous_level in sorted(section_stack.keys(), reverse=True):

        if previous_level >= level:
            continue

        parent = section_stack[previous_level]

        if parent.get("section_number"):
            return parent["section_number"]

    return None


def build_section_path(section_stack, level, current_number, current_title):
    """
    Build the full logical section path.

    Example:
        5.3 Registration and Connection Management
        5.3.2 Registration Management
        5.3.2.1 General
    """

    path = []

    for previous_level in sorted(section_stack.keys()):

        if previous_level >= level:
            continue

        parent = section_stack[previous_level]

        if not parent.get("section_number"):
            continue

        path.append(f"{parent['section_number']} " f"{parent['title']}")

    if current_number:
        path.append(f"{current_number} " f"{current_title}")
    elif current_title:
        path.append(current_title)

    return path


# ============================================================================
# TABLE EXTRACTION
# ============================================================================


def table_to_text(rows) -> str:
    """
    Convert a DOCX table into searchable text.

    When the first row behaves like a header, subsequent rows are represented
    as key/value pairs to improve BM25 and embedding retrieval.
    """

    cleaned_rows = []

    for row in rows:

        cleaned = [clean_text(cell) for cell in row if clean_text(cell)]

        if cleaned:
            cleaned_rows.append(cleaned)

    if not cleaned_rows:
        return ""

    header = cleaned_rows[0]

    if len(cleaned_rows) > 1 and len(header) > 1:

        lines = []

        for row in cleaned_rows[1:]:

            pairs = []

            for index, value in enumerate(row):

                if index < len(header):

                    pairs.append(f"{header[index]}: {value}")

                else:

                    pairs.append(value)

            if pairs:
                lines.append(" | ".join(pairs))

        return "\n".join(lines)

    return "\n".join(" | ".join(row) for row in cleaned_rows)


def extract_table(table: Table, table_index: int, current_section: dict):
    """Extract a table and preserve its parent section context."""

    rows = []

    for row in table.rows:

        cells = [clean_text(cell.text) for cell in row.cells]

        rows.append(cells)

    return {
        "table_index": table_index,
        "rows": rows,
        "text": table_to_text(rows),
        "section_number": (
            current_section.get("section_number") if current_section else "Cover"
        ),
        "section_title": (
            current_section.get("title") if current_section else "Document Metadata"
        ),
        "parent_section": (
            current_section.get("parent_section") if current_section else None
        ),
        "section_path": (
            current_section.get("section_path", []) if current_section else ["Cover"]
        ),
        "section_type": (
            current_section.get("section_type") if current_section else "cover"
        ),
    }


# ============================================================================
# DOCUMENT EXTRACTION
# ============================================================================


def extract_document(file_path: Path) -> dict:
    """
    Extract a 3GPP DOCX into structured JSON.

    Preserves:

        - cover/document metadata
        - section numbers and titles
        - section hierarchy
        - parent sections
        - section paths
        - Annexes
        - paragraphs
        - tables
        - table -> section relationship
        - specification/release/version metadata

    Page numbers are intentionally not inferred from DOCX.
    """

    document = Document(file_path)

    # ------------------------------------------------------------------------
    # COVER
    # ------------------------------------------------------------------------

    cover_section = {
        "section_number": "Cover",
        "title": "Document Metadata",
        "level": 0,
        "parent_section": None,
        "section_path": ["Cover"],
        "section_type": "cover",
        "paragraph_index": -1,
        "content": [],
        "tables": [],
        "text": "",
    }

    sections = [cover_section]

    current_section = cover_section

    section_stack = {0: cover_section}

    tables = []

    paragraph_index = 0
    table_index = 0

    # ------------------------------------------------------------------------
    # ONE DOCUMENT-BODY PASS
    #
    # This is the important improvement:
    # paragraphs and tables are processed in the real DOCX order.
    # Therefore a table immediately following a heading is attached to that
    # section correctly.
    # ------------------------------------------------------------------------

    for child in document.element.body.iterchildren():

        # Ignore section properties.
        if child.tag.endswith("}sectPr"):
            continue

        # ====================================================================
        # PARAGRAPH
        # ====================================================================

        if child.tag.endswith("}p"):

            paragraph = Paragraph(child, document)

            text = clean_text(paragraph.text)

            current_index = paragraph_index

            paragraph_index += 1

            if not text:
                continue

            style = paragraph.style.name or ""

            # Skip TOC paragraphs.
            if style.lower().startswith("toc"):
                continue

            if text.lower() == "contents":
                continue

            # ---------------------------------------------------------------
            # HEADING
            # ---------------------------------------------------------------

            if is_heading(paragraph):

                level = get_heading_level(paragraph)

                section_number, title = extract_section_info(text)

                section_type = detect_section_type(section_number, title)

                # Annexes are logical top-level sections.
                if section_type == "annex":

                    effective_level = 1

                    parent_section = None

                    section_stack = {}

                else:

                    effective_level = level

                    parent_section = build_parent_section(
                        section_stack, effective_level
                    )

                section_path = build_section_path(
                    section_stack, effective_level, section_number, title
                )

                current_section = {
                    "section_number": section_number,
                    "title": title,
                    "level": effective_level,
                    "parent_section": parent_section,
                    "section_path": section_path,
                    "section_type": section_type,
                    "paragraph_index": current_index,
                    "content": [],
                    "tables": [],
                    "text": "",
                }

                sections.append(current_section)

                section_stack[effective_level] = current_section

                # Remove deeper hierarchy levels.
                deeper_levels = [key for key in section_stack if key > effective_level]

                for key in deeper_levels:
                    del section_stack[key]

            # ---------------------------------------------------------------
            # NORMAL CONTENT
            # ---------------------------------------------------------------

            else:

                current_section["content"].append(
                    {"text": text, "style": style, "paragraph_index": current_index}
                )

        # ====================================================================
        # TABLE
        # ====================================================================

        elif child.tag.endswith("}tbl"):

            table = Table(child, document)

            record = extract_table(table, table_index, current_section)

            tables.append(record)

            current_section["tables"].append(table_index)

            table_index += 1

    # ------------------------------------------------------------------------
    # BUILD SECTION TEXT
    # ------------------------------------------------------------------------

    for section in sections:

        section["text"] = "\n".join(item["text"] for item in section.get("content", []))

    # ------------------------------------------------------------------------
    # SPECIFICATION
    # ------------------------------------------------------------------------

    spec_match = re.search(r"(?<!\d)(\d{5})(?!\d)", file_path.stem)

    if spec_match:

        specification_number = spec_match.group(1)

    else:

        specification_number = file_path.stem

    if len(specification_number) == 5 and specification_number.isdigit():

        specification = f"TS {specification_number[:2]}." f"{specification_number[2:]}"

    else:

        specification = specification_number

    # ------------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------------

    statistics = {
        "total_paragraphs": len(document.paragraphs),
        "total_sections": len(sections),
        "total_tables": len(tables),
        "numbered_sections": sum(
            1 for section in sections if section.get("section_type") == "numbered"
        ),
        "unnumbered_sections": sum(
            1 for section in sections if section.get("section_type") == "unnumbered"
        ),
        "annex_sections": sum(
            1 for section in sections if section.get("section_type") == "annex"
        ),
        "cover_sections": sum(
            1 for section in sections if section.get("section_type") == "cover"
        ),
    }

    # ------------------------------------------------------------------------
    # FINAL OBJECT
    # ------------------------------------------------------------------------

    return {
        "document": {
            "filename": file_path.name,
            "specification": specification,
            "release": "19",
            "version": "19.6.0",
            "source_type": "DOCX",
        },
        "statistics": statistics,
        "sections": sections,
        "tables": tables,
    }


# ============================================================================
# MAIN
# ============================================================================


def main():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.docx"))

    if not files:

        print("No DOCX files found.")

        print(f"Expected directory: " f"{RAW_DIR.resolve()}")

        return

    print("=" * 80)
    print("3GPP DOCUMENT INGESTION")
    print("=" * 80)

    for file_path in files:

        print(f"\nProcessing: " f"{file_path.name}")

        try:

            result = extract_document(file_path)

            output_file = PROCESSED_DIR / f"{file_path.stem}.json"

            with open(output_file, "w", encoding="utf-8") as file:

                json.dump(result, file, indent=2, ensure_ascii=False)

            print(f"  Sections      : " f"{result['statistics']['total_sections']}")

            print(f"  Numbered      : " f"{result['statistics']['numbered_sections']}")

            print(
                f"  Unnumbered    : " f"{result['statistics']['unnumbered_sections']}"
            )

            print(f"  Tables        : " f"{result['statistics']['total_tables']}")

            print(f"  Annexes       : " f"{result['statistics']['annex_sections']}")

            print(f"  Cover         : " f"{result['statistics']['cover_sections']}")

            print(f"  Specification : " f"{result['document']['specification']}")

            print(f"  Release       : " f"{result['document']['release']}")

            print(f"  Version       : " f"{result['document']['version']}")

            print(f"  Output        : " f"{output_file}")

        except Exception as error:

            print(f"  ERROR: {error}")

    print("\n" + "=" * 80)

    print("INGESTION COMPLETE")

    print("=" * 80)


if __name__ == "__main__":
    main()
