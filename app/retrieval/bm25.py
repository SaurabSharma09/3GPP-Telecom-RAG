from pathlib import Path
import json
import pickle
import re

from rank_bm25 import BM25Okapi


# ============================================================================
# DIRECTORIES
# ============================================================================

CHUNK_DIR = Path("data/processed/chunks")
VECTOR_DIR = Path("data/processed/vector_store")


# ============================================================================
# TOKENIZATION
# ============================================================================


def tokenize(text: str):
    """
    Tokenize technical 3GPP text.

    Keeps identifiers such as:
        AMF
        N11
        N2
        S-NSSAI
        NG-RAN
        Nsmf_PDUSession_CreateSMContext

    Lowercasing is intentional because BM25 retrieval is case-insensitive.
    """

    if not text:
        return []

    text = str(text).lower()

    tokens = re.findall(r"[a-zA-Z0-9]+(?:[-_/][a-zA-Z0-9]+)*", text)

    return tokens


# ============================================================================
# CHUNK LOADING
# ============================================================================


def load_chunks():

    if not CHUNK_DIR.exists():

        raise FileNotFoundError(
            f"Chunk directory does not exist: " f"{CHUNK_DIR.resolve()}"
        )

    files = sorted(CHUNK_DIR.glob("*_chunks.json"))

    if not files:

        raise FileNotFoundError(
            f"No *_chunks.json files found in " f"{CHUNK_DIR.resolve()}"
        )

    all_chunks = []

    print(f"Found {len(files)} chunk file(s).")

    for file_path in files:

        print(f"Loading: {file_path.name}")

        with open(file_path, "r", encoding="utf-8") as file:

            data = json.load(file)

        chunks = data.get("chunks", [])

        if not chunks:

            print(f"  WARNING: {file_path.name} " f"contains no chunks.")

            continue

        all_chunks.extend(chunks)

        print(f"  Chunks: {len(chunks)}")

    if not all_chunks:

        raise ValueError(
            "No chunks were loaded from the processed " "3GPP chunk files."
        )

    return all_chunks


# ============================================================================
# VALIDATION
# ============================================================================


def validate_chunks(chunks):
    """
    Validate the chunk data before constructing BM25.

    The BM25 corpus and the metadata list must have exactly the same
    ordering because retrieval returns indices into this list.
    """

    errors = []

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):

            errors.append(f"Chunk {index} is not a dictionary.")

            continue

        if not chunk.get("text", "").strip():

            errors.append(f"Chunk {index} has empty text.")

        metadata = chunk.get("metadata")

        if not isinstance(metadata, dict):

            errors.append(f"Chunk {index} has invalid metadata.")

            continue

        required_fields = [
            "specification",
            "section_number",
            "section_title",
            "chunk_type",
        ]

        for field in required_fields:

            if field not in metadata:

                errors.append(f"Chunk {index} missing " f"metadata field: {field}")

    if errors:

        print("\nBM25 validation errors:")

        for error in errors[:20]:

            print(f"  ERROR: {error}")

        if len(errors) > 20:

            print(f"  ...and {len(errors) - 20} more.")

        raise ValueError(
            "Chunk validation failed. " "Fix chunking before rebuilding BM25."
        )


# ============================================================================
# BM25 BUILDING
# ============================================================================


def build_bm25(chunks):

    print(f"\nBuilding BM25 index for " f"{len(chunks)} chunks...")

    validate_chunks(chunks)

    tokenized_corpus = []

    valid_chunks = []

    empty_token_documents = 0

    for chunk in chunks:

        tokens = tokenize(chunk["text"])

        if not tokens:

            empty_token_documents += 1

            continue

        tokenized_corpus.append(tokens)

        valid_chunks.append(chunk)

    if empty_token_documents:

        print(
            f"WARNING: skipped " f"{empty_token_documents} " f"empty-token documents."
        )

    if not tokenized_corpus:

        raise ValueError("BM25 corpus is empty after tokenization.")

    print(f"Indexed chunks: " f"{len(valid_chunks)}")

    # ------------------------------------------------------------------------
    # Chunk type statistics
    # ------------------------------------------------------------------------

    text_chunks = sum(
        1 for chunk in valid_chunks if chunk["metadata"].get("chunk_type") == "text"
    )

    table_chunks = sum(
        1 for chunk in valid_chunks if chunk["metadata"].get("chunk_type") == "table"
    )

    print(f"Text chunks: {text_chunks}")

    print(f"Table chunks: {table_chunks}")

    # ------------------------------------------------------------------------
    # Build BM25
    # ------------------------------------------------------------------------

    bm25 = BM25Okapi(tokenized_corpus)

    # ------------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------------

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    index_path = VECTOR_DIR / "bm25.pkl"

    metadata_path = VECTOR_DIR / "bm25_chunks.pkl"

    # ------------------------------------------------------------------------
    # Save BM25 index
    # ------------------------------------------------------------------------

    with open(index_path, "wb") as file:

        pickle.dump(bm25, file)

    # IMPORTANT:
    # Save exactly the same chunk ordering used by tokenized_corpus.
    with open(metadata_path, "wb") as file:

        pickle.dump(valid_chunks, file)

    print("\nBM25 index created!")

    print(f"Documents: " f"{len(valid_chunks)}")

    print(f"Index: " f"{index_path}")

    print(f"Metadata: " f"{metadata_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():

    print("=" * 80)
    print("3GPP BM25 INDEXING")
    print("=" * 80)

    chunks = load_chunks()

    build_bm25(chunks)

    print("\n" + "=" * 80)

    print("BM25 INDEXING COMPLETE")

    print("=" * 80)


if __name__ == "__main__":
    main()
