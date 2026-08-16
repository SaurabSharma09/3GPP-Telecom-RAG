from pathlib import Path
import json
import pickle
import re

from rank_bm25 import BM25Okapi


CHUNK_DIR = Path("data/processed/chunks")
VECTOR_DIR = Path("data/processed/vector_store")


def tokenize(text: str):
    """
    Tokenize technical text while preserving
    identifiers such as AMF, N11, S-NSSAI, etc.
    """

    text = text.lower()

    # Keep alphanumeric technical identifiers.
    tokens = re.findall(
        r"[a-zA-Z0-9]+(?:[-_/][a-zA-Z0-9]+)*",
        text
    )

    return tokens


def load_chunks():

    all_chunks = []

    files = sorted(
        CHUNK_DIR.glob("*_chunks.json")
    )

    for file_path in files:

        print(
            f"Loading: {file_path.name}"
        )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        all_chunks.extend(
            data["chunks"]
        )

    return all_chunks


def build_bm25(chunks):

    print(
        f"Building BM25 index for "
        f"{len(chunks)} chunks..."
    )

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    index_path = (
        VECTOR_DIR / "bm25.pkl"
    )

    metadata_path = (
        VECTOR_DIR / "bm25_chunks.pkl"
    )

    with open(
        index_path,
        "wb"
    ) as file:

        pickle.dump(
            bm25,
            file
        )

    with open(
        metadata_path,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print("\nBM25 index created!")

    print(
        f"Documents: {len(chunks)}"
    )

    print(
        f"Index: {index_path}"
    )

    print(
        f"Metadata: {metadata_path}"
    )


def main():

    print("=" * 80)
    print("3GPP BM25 INDEXING")
    print("=" * 80)

    chunks = load_chunks()

    build_bm25(
        chunks
    )

    print("\n" + "=" * 80)
    print("BM25 INDEXING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()