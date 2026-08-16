from pathlib import Path
import json
import re

from docx import Document


RAW_DIR = Path("data/raw/3gpp")
PROCESSED_DIR = Path("data/processed/3gpp")


def clean_text(text: str) -> str:
    """Clean whitespace without removing meaningful content."""

    text = text.replace("\xa0", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_section_info(text: str):
    """
    Extract an explicit 3GPP section number and title.

    Examples:
        1 Scope
        4.2 Architecture reference model
        4.2.2 Network Functions and entities
        4.2.5a Radio Capabilities Signalling optimisation

    Returns:
        (section_number, title)

    If the heading does not contain an explicit section
    number, section_number is None.
    """

    text = clean_text(text)

    pattern = (
        r"^("
        r"\d+"
        r"(?:\.\d+)*"
        r"[A-Za-z]?"
        r")"
        r"\s+"
        r"(.+)$"
    )

    match = re.match(
        pattern,
        text
    )

    if not match:
        return None, text

    return (
        match.group(1),
        match.group(2).strip()
    )


def is_heading(paragraph) -> bool:
    """Check whether paragraph is a real document heading."""

    style = paragraph.style.name

    return style.startswith(
        "Heading "
    )


def get_heading_level(paragraph) -> int:
    """Convert Heading 1 → 1, Heading 2 → 2, etc."""

    match = re.match(
        r"Heading\s+(\d+)",
        paragraph.style.name
    )

    if match:
        return int(
            match.group(1)
        )

    return 0


def build_parent_section(
    section_stack,
    level
):
    """
    Find the nearest numbered parent section.

    Example:

        4
        └── 4.2
            └── 4.2.1
                └── unnumbered heading

    The unnumbered heading receives:

        parent_section = 4.2.1
    """

    for previous_level in range(
        level - 1,
        0,
        -1
    ):

        parent = section_stack.get(
            previous_level
        )

        if parent is None:
            continue

        if parent.get(
            "section_number"
        ):

            return parent[
                "section_number"
            ]

    return None


def extract_document(
    file_path: Path
) -> dict:

    document = Document(
        file_path
    )

    sections = []

    current_section = None

    started = False

    # Stores the latest section at each
    # heading level.
    section_stack = {}

    for index, paragraph in enumerate(
        document.paragraphs
    ):

        text = clean_text(
            paragraph.text
        )

        if not text:
            continue

        style = paragraph.style.name

        # --------------------------------------------------
        # Ignore Table of Contents
        # --------------------------------------------------

        if style.lower().startswith(
            "toc"
        ):
            continue

        # Ignore Contents heading
        if text.lower() == "contents":
            continue

        # --------------------------------------------------
        # Start from Foreword
        # --------------------------------------------------

        if not started:

            if text.lower() == "foreword":
                started = True
            else:
                continue

        # --------------------------------------------------
        # Heading
        # --------------------------------------------------

        if is_heading(paragraph):

            level = get_heading_level(
                paragraph
            )

            section_number, title = (
                extract_section_info(
                    text
                )
            )

            # Find nearest numbered parent.
            parent_section = (
                build_parent_section(
                    section_stack,
                    level
                )
            )

            current_section = {

                "section_number":
                    section_number,

                "title":
                    title,

                "level":
                    level,

                "parent_section":
                    parent_section,

                "paragraph_index":
                    index,

                "content":
                    []
            }

            sections.append(
                current_section
            )

            # Update section stack.
            section_stack[level] = (
                current_section
            )

            # Remove deeper levels because
            # they no longer belong to the
            # current hierarchy.
            deeper_levels = [
                key
                for key in section_stack
                if key > level
            ]

            for key in deeper_levels:
                del section_stack[key]

        # --------------------------------------------------
        # Content
        # --------------------------------------------------

        else:

            if current_section is not None:

                current_section[
                    "content"
                ].append({

                    "text": text,

                    "style": style,

                    "paragraph_index":
                        index
                })

    # ------------------------------------------------------
    # Build section text
    # ------------------------------------------------------

    for section in sections:

        section["text"] = "\n".join(
            item["text"]
            for item in section[
                "content"
            ]
        )

    # ------------------------------------------------------
    # Extract tables
    # ------------------------------------------------------

    tables = []

    for table_index, table in enumerate(
        document.tables
    ):

        rows = []

        for row in table.rows:

            cells = [
                clean_text(
                    cell.text
                )
                for cell in row.cells
            ]

            rows.append(
                cells
            )

        tables.append({

            "table_index":
                table_index,

            "rows":
                rows
        })

    # ------------------------------------------------------
    # Determine specification
    # ------------------------------------------------------

    spec_match = re.search(
        r"(\d{5})",
        file_path.stem
    )

    specification = (
        spec_match.group(1)
        if spec_match
        else file_path.stem
    )

    # ------------------------------------------------------
    # Final document object
    # ------------------------------------------------------

    return {

        "document": {

            "filename":
                file_path.name,

            "specification":
                f"TS {specification[:2]}."
                f"{specification[2:]}",

            "release":
                "19",

            "version":
                "19.6.0"
        },

        "statistics": {

            "total_paragraphs":
                len(
                    document.paragraphs
                ),

            "total_sections":
                len(sections),

            "total_tables":
                len(tables)
        },

        "sections":
            sections,

        "tables":
            tables
    }


def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        RAW_DIR.glob(
            "*.docx"
        )
    )

    if not files:

        print(
            "No DOCX files found."
        )

        return

    print("=" * 80)
    print(
        "3GPP DOCUMENT INGESTION"
    )
    print("=" * 80)

    for file_path in files:

        print(
            f"\nProcessing: "
            f"{file_path.name}"
        )

        try:

            result = extract_document(
                file_path
            )

            output_name = (
                file_path.stem
                + ".json"
            )

            output_file = (
                PROCESSED_DIR
                / output_name
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    result,
                    file,
                    indent=2,
                    ensure_ascii=False
                )

            print(
                f"  Sections : "
                f"{result['statistics']['total_sections']}"
            )

            print(
                f"  Tables   : "
                f"{result['statistics']['total_tables']}"
            )

            # Show how many sections have
            # explicit section numbers.
            numbered = sum(
                1
                for section
                in result["sections"]
                if section[
                    "section_number"
                ] is not None
            )

            unnumbered = (
                len(result["sections"])
                - numbered
            )

            print(
                f"  Numbered : "
                f"{numbered}"
            )

            print(
                f"  Unnumbered: "
                f"{unnumbered}"
            )

            print(
                f"  Output   : "
                f"{output_file}"
            )

        except Exception as error:

            print(
                f"  ERROR: "
                f"{error}"
            )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()