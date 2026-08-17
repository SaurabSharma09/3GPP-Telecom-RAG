from pathlib import Path
import json
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================================
# DIRECTORIES
# ============================================================================

CHUNK_DIR = Path("data/processed/chunks")
VECTOR_DIR = Path("data/processed/vector_store")


# ============================================================================
# EMBEDDING MODEL
# ============================================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================================
# LOAD CHUNKS
# ============================================================================


def load_chunks():
    """Load all processed 3GPP chunks in deterministic order."""

    if not CHUNK_DIR.exists():
        raise FileNotFoundError(f"Chunk directory not found: " f"{CHUNK_DIR.resolve()}")

    files = sorted(CHUNK_DIR.glob("*_chunks.json"))

    if not files:
        raise FileNotFoundError(
            f"No *_chunks.json files found in " f"{CHUNK_DIR.resolve()}"
        )

    all_chunks = []

    for file_path in files:

        print(f"Loading: {file_path.name}")

        with open(file_path, "r", encoding="utf-8") as file:

            data = json.load(file)

        file_chunks = data.get("chunks", [])

        if not file_chunks:
            print("  WARNING: no chunks found.")
            continue

        all_chunks.extend(file_chunks)

        print(f"  Chunks: {len(file_chunks)}")

    if not all_chunks:
        raise ValueError("No chunks were loaded.")

    print(f"\nTotal chunks loaded: " f"{len(all_chunks)}")

    return all_chunks


# ============================================================================
# VALIDATION
# ============================================================================


def validate_chunks(chunks):
    """
    Validate the final chunk set before embedding.

    This is important because the FAISS vector position and the metadata
    position must remain exactly aligned.
    """

    errors = []

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            errors.append(f"Chunk {index} is not a dictionary.")
            continue

        text = chunk.get("text", "")

        if not isinstance(text, str) or not text.strip():

            errors.append(f"Chunk {index} has empty text.")

        metadata = chunk.get("metadata", {})

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

                errors.append(f"Chunk {index} missing " f"metadata field '{field}'.")

    if errors:

        print("\n❌ VECTOR INDEX VALIDATION FAILED")

        for error in errors[:20]:

            print(f"  ERROR: {error}")

        if len(errors) > 20:

            print(f"  ...and {len(errors) - 20} more.")

        raise ValueError(
            "Chunk validation failed. " "Fix chunking before building FAISS."
        )


# ============================================================================
# BUILD VECTOR INDEX
# ============================================================================


def build_index(chunks):

    validate_chunks(chunks)

    print(f"\nLoading embedding model: " f"{MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # ------------------------------------------------------------------------
    # Prepare texts
    # ------------------------------------------------------------------------

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for " f"{len(texts)} chunks...")

    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # ------------------------------------------------------------------------
    # Validate embeddings
    # ------------------------------------------------------------------------

    if embeddings.ndim != 2:

        raise ValueError(f"Unexpected embedding shape: " f"{embeddings.shape}")

    if embeddings.shape[0] != len(chunks):

        raise ValueError(
            "Embedding count does not match "
            "chunk count.\n"
            f"Chunks: {len(chunks)}\n"
            f"Embeddings: {embeddings.shape[0]}"
        )

    if not np.isfinite(embeddings).all():

        raise ValueError("Embedding matrix contains " "NaN or infinite values.")

    dimension = embeddings.shape[1]

    print(f"Embedding dimension: " f"{dimension}")

    # ------------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------------
    #
    # Embeddings are L2-normalized above.
    # Inner product therefore corresponds to cosine similarity.
    #

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    index_path = VECTOR_DIR / "3gpp.index"

    metadata_path = VECTOR_DIR / "chunks.pkl"

    faiss.write_index(index, str(index_path))

    # IMPORTANT:
    # Save exactly the same ordering as the vectors added to FAISS.
    with open(metadata_path, "wb") as file:

        pickle.dump(chunks, file)

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    text_chunks = sum(
        1 for chunk in chunks if chunk["metadata"].get("chunk_type") == "text"
    )

    table_chunks = sum(
        1 for chunk in chunks if chunk["metadata"].get("chunk_type") == "table"
    )

    print("\nVector store created!")

    print(f"Vectors : {index.ntotal}")

    print(f"Chunks  : {len(chunks)}")

    print(f"Text    : {text_chunks}")

    print(f"Tables  : {table_chunks}")

    print(f"Dimension: {dimension}")

    print(f"Index   : {index_path}")

    print(f"Metadata: {metadata_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():

    print("=" * 80)
    print("3GPP VECTOR INDEXING")
    print("=" * 80)

    chunks = load_chunks()

    build_index(chunks)

    print("\n" + "=" * 80)

    print("VECTOR INDEXING COMPLETE")

    print("=" * 80)


if __name__ == "__main__":
    main()
