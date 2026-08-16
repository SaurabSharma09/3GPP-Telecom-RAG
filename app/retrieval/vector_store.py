from pathlib import Path
import json
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CHUNK_DIR = Path("data/processed/chunks")
VECTOR_DIR = Path("data/processed/vector_store")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    """Load chunks from all processed 3GPP documents."""

    all_chunks = []

    files = sorted(
        CHUNK_DIR.glob("*_chunks.json")
    )

    if not files:
        raise FileNotFoundError(
            "No chunk files found."
        )

    for file_path in files:

        print(f"Loading: {file_path.name}")

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        all_chunks.extend(
            data["chunks"]
        )

    print(
        f"Total chunks loaded: {len(all_chunks)}"
    )

    return all_chunks


def build_index(chunks):

    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "Generating embeddings..."
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    print(
        f"Embedding dimension: {dimension}"
    )

    # Cosine similarity through inner product
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    index_path = (
        VECTOR_DIR / "3gpp.index"
    )

    metadata_path = (
        VECTOR_DIR / "chunks.pkl"
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    with open(
        metadata_path,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print("\nVector store created!")

    print(
        f"Vectors : {index.ntotal}"
    )

    print(
        f"Dimension: {dimension}"
    )

    print(
        f"Index   : {index_path}"
    )

    print(
        f"Metadata: {metadata_path}"
    )


def main():

    print("=" * 80)
    print("3GPP VECTOR INDEXING")
    print("=" * 80)

    chunks = load_chunks()

    build_index(
        chunks
    )

    print("\n" + "=" * 80)
    print("VECTOR INDEXING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()