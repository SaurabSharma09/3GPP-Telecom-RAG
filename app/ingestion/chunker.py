from pathlib import Path
import json
import re


# ============================================================================
# DIRECTORIES
# ============================================================================

PROCESSED_DIR = Path("data/processed/3gpp")
CHUNK_DIR = Path("data/processed/chunks")


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_CHARS = 3000
MAX_CHARS = 4500
OVERLAP_CHARS = 300


# ============================================================================
# TEXT UTILITIES
# ============================================================================


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving technical content."""

    if not text:
        return ""

    text = str(text)

    text = text.replace("\xa0", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_path(path):
    """Return a cleaned section path."""

    if not path:
        return []

    return [clean_text(item) for item in path if clean_text(item)]


# ============================================================================
# SECTION CONTEXT
# ============================================================================


def section_label(section):
    """Build a compact searchable section label."""

    number = clean_text(section.get("section_number", ""))

    title = clean_text(section.get("title", ""))

    if number and title:
        return f"{number} {title}"

    return number or title or "Unspecified Section"


def build_context_prefix(section):
    """
    Add section identity to indexed text.

    This helps both BM25 and embeddings understand that a chunk belongs to,
    for example, 6.2.2 SMF or 5.3.2 Registration Management.
    """

    label = section_label(section)

    path = normalize_path(section.get("section_path", []))

    prefix = f"3GPP Section: {label}\n"

    if path:
        prefix += "Section Path: " + " > ".join(path) + "\n"

    return prefix


def build_table_prefix(table):
    """Build searchable context for a table."""

    specification = clean_text(table.get("specification", ""))

    section_number = clean_text(table.get("section_number", ""))

    section_title = clean_text(table.get("section_title", ""))

    section_path = normalize_path(table.get("section_path", []))

    if section_number and section_title:
        label = f"{section_number} " f"{section_title}"
    else:
        label = section_title or section_number or "Unspecified Section"

    prefix = f"3GPP Specification: {specification}\n" f"Table Section: {label}\n"

    if section_path:
        prefix += "Section Path: " + " > ".join(section_path) + "\n"

    return prefix


# ============================================================================
# SMART SPLITTING
# ============================================================================


def split_long_text(text, max_chars=MAX_CHARS):
    """
    Split long text at sentence boundaries where possible,
    then word boundaries if necessary.
    """

    text = clean_text(text)

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)

    pieces = []
    current = ""

    for sentence in sentences:

        sentence = clean_text(sentence)

        if not sentence:
            continue

        candidate = f"{current} {sentence}" if current else sentence

        if len(candidate) <= max_chars:

            current = candidate
            continue

        if current:
            pieces.append(current)
            current = ""

        if len(sentence) > max_chars:

            words = sentence.split()
            buffer = ""

            for word in words:

                candidate_word = f"{buffer} {word}" if buffer else word

                if len(candidate_word) <= max_chars:

                    buffer = candidate_word

                else:

                    if buffer:
                        pieces.append(buffer)

                    buffer = word

            if buffer:
                current = buffer

        else:

            current = sentence

    if current:
        pieces.append(current)

    return pieces


# ============================================================================
# SECTION CHUNKING
# ============================================================================


def split_section(section):
    """
    Split one section into chunks.

    Small sections remain whole.
    Large sections are split without crossing into another section.
    """

    prefix = build_context_prefix(section)

    # Reserve space for the metadata prefix so final indexed text never
    # exceeds MAX_CHARS.
    payload_limit = max(1000, MAX_CHARS - len(prefix) - 2)

    target_limit = min(TARGET_CHARS, payload_limit)

    paragraphs = []

    for item in section.get("content", []):

        text = clean_text(item.get("text", ""))

        if not text:
            continue

        paragraphs.extend(split_long_text(text, payload_limit))

    # Backward compatibility.
    if not paragraphs:

        fallback = clean_text(section.get("text", ""))

        if fallback:

            paragraphs.extend(split_long_text(fallback, payload_limit))

    if not paragraphs:
        return [], payload_limit

    chunks = []
    current = ""

    for paragraph in paragraphs:

        candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(candidate) <= target_limit:

            current = candidate
            continue

        if current:
            chunks.append(current)

        current = paragraph

    if current:
        chunks.append(current)

    return chunks, payload_limit


def add_overlap(chunks, max_size=None):
    """Add modest overlap only between chunks of the same section."""

    if max_size is None:
        max_size = MAX_CHARS

    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]

    for index in range(1, len(chunks)):

        previous = chunks[index - 1]
        current = chunks[index]

        overlap = previous[-OVERLAP_CHARS:]

        if " " in overlap:
            overlap = overlap.split(" ", 1)[1]

        combined = overlap + "\n\n" + current

        if len(combined) > max_size:
            combined = current

        result.append(combined)

    return result


# ============================================================================
# TABLE CHUNKING
# ============================================================================


def create_table_chunks(document):
    """
    Convert extracted tables into searchable chunks while preserving
    parent section metadata.
    """

    chunks = []

    doc_meta = document["document"]

    for table in document.get("tables", []):

        table_text = clean_text(table.get("text", ""))

        if not table_text:
            continue

        prefix = build_table_prefix(
            {**table, "specification": doc_meta.get("specification")}
        )

        payload_limit = max(1000, MAX_CHARS - len(prefix) - 2)

        pieces = split_long_text(table_text, payload_limit)

        for local_index, piece in enumerate(pieces):

            metadata = {
                "specification": doc_meta.get("specification"),
                "release": doc_meta.get("release"),
                "version": doc_meta.get("version"),
                "source_file": doc_meta.get("filename"),
                "section_number": table.get("section_number"),
                "section_title": table.get("section_title"),
                "parent_section": table.get("parent_section"),
                "section_path": normalize_path(table.get("section_path", [])),
                "section_type": table.get("section_type"),
                "section_level": None,
                "paragraph_index": None,
                "local_chunk_index": local_index,
                "chunk_type": "table",
                "table_index": table.get("table_index"),
            }

            chunks.append(
                {
                    "chunk_id": (
                        f"{doc_meta['specification']}"
                        f"_TABLE_"
                        f"{table.get('table_index', 0):04d}_"
                        f"{local_index:02d}"
                    ),
                    "text": (prefix + "\n" + piece),
                    "metadata": metadata,
                }
            )

    return chunks


# ============================================================================
# TEXT CHUNK CREATION
# ============================================================================


def create_text_chunks(document):

    chunks = []

    doc_meta = document["document"]

    chunk_id = 0

    for section in document.get("sections", []):

        raw_chunks, payload_limit = split_section(section)

        raw_chunks = add_overlap(raw_chunks, max_size=payload_limit)

        prefix = build_context_prefix(section)

        for local_index, raw_text in enumerate(raw_chunks):

            raw_text = clean_text(raw_text)

            if not raw_text:
                continue

            searchable_text = prefix + "\n" + raw_text

            metadata = {
                "specification": doc_meta.get("specification"),
                "release": doc_meta.get("release"),
                "version": doc_meta.get("version"),
                "source_file": doc_meta.get("filename"),
                "section_number": section.get("section_number"),
                "section_title": section.get("title"),
                "parent_section": section.get("parent_section"),
                "section_path": normalize_path(section.get("section_path", [])),
                "section_type": section.get("section_type", "unnumbered"),
                "section_level": section.get("level"),
                "paragraph_index": section.get("paragraph_index"),
                "local_chunk_index": local_index,
                "chunk_type": "text",
                "table_index": None,
            }

            chunks.append(
                {
                    "chunk_id": (f"{doc_meta['specification']}_" f"{chunk_id:06d}"),
                    "text": searchable_text,
                    "metadata": metadata,
                }
            )

            chunk_id += 1

    return chunks


# ============================================================================
# ALL CHUNKS
# ============================================================================


def create_chunks(document):

    text_chunks = create_text_chunks(document)

    table_chunks = create_table_chunks(document)

    return text_chunks + table_chunks


# ============================================================================
# VALIDATION
# ============================================================================


def validate_chunks(document, chunks):
    """
    Validate chunk quality without making assumptions that every
    specification contains the same section numbers.
    """

    errors = []
    warnings = []

    source_sections = document.get("sections", [])

    source_tables = document.get("tables", [])

    # ------------------------------------------------------------------------
    # Check every source section with actual content/tables
    # ------------------------------------------------------------------------

    chunks_by_section = {}

    for chunk in chunks:

        section_number = chunk["metadata"].get("section_number")

        chunks_by_section.setdefault(section_number, []).append(chunk)

    for section in source_sections:

        section_number = section.get("section_number")

        direct_content = bool(section.get("content"))

        direct_tables = bool(section.get("tables"))

        # A parent heading can legitimately have no direct chunk because
        # its child subsections contain all technical content.
        if (
            section_number
            and (direct_content or direct_tables)
            and section_number not in chunks_by_section
        ):

            errors.append(
                f"Section {section_number} has " f"direct content/tables but no chunks."
            )

    # ------------------------------------------------------------------------
    # Size checks
    # ------------------------------------------------------------------------

    oversized = [chunk for chunk in chunks if len(chunk.get("text", "")) > MAX_CHARS]

    if oversized:

        errors.append(f"{len(oversized)} chunks exceed " f"MAX_CHARS={MAX_CHARS}.")

    # ------------------------------------------------------------------------
    # Required metadata
    # ------------------------------------------------------------------------

    required_fields = ["specification", "section_number", "section_title", "chunk_type"]

    for index, chunk in enumerate(chunks):

        metadata = chunk.get("metadata", {})

        if not isinstance(metadata, dict):

            errors.append(f"Chunk {index} has invalid metadata.")

            continue

        for field in required_fields:

            if field not in metadata:

                errors.append(f"Chunk {index} missing " f"metadata field '{field}'.")

    # ------------------------------------------------------------------------
    # Table validation
    # ------------------------------------------------------------------------

    table_chunks = [
        chunk for chunk in chunks if chunk["metadata"].get("chunk_type") == "table"
    ]

    if source_tables and not table_chunks:

        warnings.append("Source contains tables but no " "table chunks were created.")

    # ------------------------------------------------------------------------
    # Output sanity
    # ------------------------------------------------------------------------

    if not chunks:

        errors.append("No chunks were created.")

    return {"valid": not errors, "errors": errors, "warnings": warnings}


# ============================================================================
# PROCESS ONE JSON
# ============================================================================


def process_file(json_file: Path):

    print(f"\nProcessing: {json_file.name}")

    with open(json_file, "r", encoding="utf-8") as file:

        document = json.load(file)

    chunks = create_chunks(document)

    validation = validate_chunks(document, chunks)

    if not validation["valid"]:

        print("\n❌ CHUNK VALIDATION FAILED")

        for error in validation["errors"]:

            print(f"  ERROR: {error}")

        raise ValueError("Chunk validation failed.")

    if validation["warnings"]:

        print("\n⚠ WARNINGS")

        for warning in validation["warnings"]:

            print(f"  WARNING: {warning}")

    text_count = sum(
        1 for chunk in chunks if chunk["metadata"].get("chunk_type") == "text"
    )

    table_count = sum(
        1 for chunk in chunks if chunk["metadata"].get("chunk_type") == "table"
    )

    output = {
        "document": document["document"],
        "statistics": {
            "source_sections": document["statistics"].get("total_sections", 0),
            "source_tables": document["statistics"].get("total_tables", 0),
            "total_chunks": len(chunks),
            "text_chunks": text_count,
            "table_chunks": table_count,
            "validation_passed": True,
        },
        "chunks": chunks,
    }

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    output_file = CHUNK_DIR / f"{json_file.stem}_chunks.json"

    with open(output_file, "w", encoding="utf-8") as file:

        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"  Text chunks  : {text_count}")

    print(f"  Table chunks : {table_count}")

    print(f"  Total chunks : {len(chunks)}")

    print("  Validation   : PASS")

    print(f"  Output       : {output_file}")

    return output


# ============================================================================
# MAIN
# ============================================================================


def main():

    print("=" * 80)
    print("3GPP STRUCTURE-AWARE CHUNKING")
    print("=" * 80)

    files = sorted(PROCESSED_DIR.glob("*.json"))

    if not files:

        print("No processed JSON documents found.")

        print(f"Expected directory: " f"{PROCESSED_DIR.resolve()}")

        return

    success = 0

    for json_file in files:

        try:

            process_file(json_file)

            success += 1

        except Exception as error:

            print(f"\n❌ ERROR processing " f"{json_file.name}: {error}")

    print("\n" + "=" * 80)

    print(f"CHUNKING COMPLETE — " f"{success}/{len(files)} documents processed")

    print("=" * 80)


if __name__ == "__main__":
    main()
