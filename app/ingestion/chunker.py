from pathlib import Path
import json
import re


PROCESSED_DIR = Path("data/processed/3gpp")
CHUNK_DIR = Path("data/processed/chunks")


# Target size for each chunk.
# We use characters here because our source is already structured text.
TARGET_CHARS = 4000

# Maximum chunk size before we force a split.
MAX_CHARS = 6000

# Number of characters carried from the previous chunk.
OVERLAP_CHARS = 500


def clean_text(text: str) -> str:
    """Normalize whitespace."""

    text = text.replace("\xa0", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_text(text: str):
    """
    Split section text while trying to preserve paragraph boundaries.
    """

    paragraphs = [
        clean_text(p)
        for p in text.split("\n")
        if clean_text(p)
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        # If adding the next paragraph stays reasonable,
        # keep it in the current chunk.
        candidate = (
            f"{current}\n\n{paragraph}"
            if current
            else paragraph
        )

        if len(candidate) <= TARGET_CHARS:

            current = candidate

        else:

            if current:
                chunks.append(current)

            # Very large individual paragraphs need
            # a hard split.
            if len(paragraph) > MAX_CHARS:

                start = 0

                while start < len(paragraph):

                    end = start + MAX_CHARS

                    piece = paragraph[start:end]

                    chunks.append(piece)

                    start = end

                current = ""

            else:

                current = paragraph

    if current:
        chunks.append(current)

    return chunks


def add_overlap(chunks):
    """Add limited overlap between consecutive chunks."""

    if not chunks:
        return chunks

    result = [chunks[0]]

    for i in range(1, len(chunks)):

        previous = chunks[i - 1]

        overlap = previous[-OVERLAP_CHARS:]

        current = (
            overlap
            + "\n\n"
            + chunks[i]
        )

        result.append(current)

    return result


def create_chunks(document):
    """Convert structured sections into metadata-rich chunks."""

    chunks = []

    chunk_id = 0

    metadata = document["document"]

    for section in document["sections"]:

        section_text = clean_text(
            section.get("text", "")
        )

        if not section_text:
            continue

        section_chunks = split_text(
            section_text
        )

        section_chunks = add_overlap(
            section_chunks
        )

        for local_index, text in enumerate(
            section_chunks
        ):

            text = clean_text(text)

            if not text:
                continue

            chunk = {
                "chunk_id": f"{metadata['specification']}_{chunk_id:06d}",

                "text": text,

                "metadata": {
                    "specification": metadata[
                        "specification"
                    ],

                    "release": metadata[
                        "release"
                    ],

                    "version": metadata[
                        "version"
                    ],

                    "source_file": metadata[
                        "filename"
                    ],

                   "section_number": section[
    "section_number"
],

"section_title": section[
    "title"
],

"parent_section": section.get(
    "parent_section"
),

                    "section_level": section[
                        "level"
                    ],

                    "paragraph_index": section[
                        "paragraph_index"
                    ],

                    "local_chunk_index": local_index
                }
            }

            chunks.append(chunk)

            chunk_id += 1

    return chunks


def process_file(json_file: Path):

    print(f"\nProcessing: {json_file.name}")

    with open(
        json_file,
        "r",
        encoding="utf-8"
    ) as file:

        document = json.load(file)

    chunks = create_chunks(document)

    output = {
        "document": document["document"],

        "statistics": {
            "source_sections": document[
                "statistics"
            ]["total_sections"],

            "source_tables": document[
                "statistics"
            ]["total_tables"],

            "total_chunks": len(chunks)
        },

        "chunks": chunks
    }

    CHUNK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        CHUNK_DIR
        / f"{json_file.stem}_chunks.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"  Chunks : {len(chunks)}"
    )

    print(
        f"  Output : {output_file}"
    )

    return output


def main():

    print("=" * 80)
    print("3GPP INTELLIGENT CHUNKING")
    print("=" * 80)

    files = sorted(
        PROCESSED_DIR.glob("*.json")
    )

    if not files:

        print(
            "No processed JSON documents found."
        )

        return

    for json_file in files:

        process_file(json_file)

    print("\n" + "=" * 80)
    print("CHUNKING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()