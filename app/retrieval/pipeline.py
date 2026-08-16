from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker

from app.evaluation.evidence_gate import EvidenceGate
from app.evaluation.evidence_verifier import EvidenceVerifier
from app.evaluation.semantic_verifier import (
    SemanticEvidenceVerifier
)


class RetrievalPipeline:

    def __init__(
        self,
        vector_k=20,
        bm25_k=20,
        hybrid_k=10,
        final_k=5
    ):

        print("=" * 80)
        print("INITIALIZING RETRIEVAL PIPELINE")
        print("=" * 80)


        # ==================================================
        # STAGE 1
        # HYBRID RETRIEVAL
        # ==================================================

        self.hybrid_retriever = HybridRetriever(
            vector_k=vector_k,
            bm25_k=bm25_k,
            final_k=hybrid_k
        )


        # ==================================================
        # STAGE 2
        # CROSS-ENCODER RERANKING
        # ==================================================

        self.reranker = Reranker()


        # ==================================================
        # STAGE 3
        # EVIDENCE GATE
        # ==================================================

        self.evidence_gate = EvidenceGate(

            min_score=1.0,

            min_margin=0.5,

            min_keyword_coverage=0.5
        )


        # ==================================================
        # STAGE 4
        # LEXICAL EVIDENCE VERIFIER
        # ==================================================

        self.evidence_verifier = EvidenceVerifier(

            min_supporting_terms=0.5
        )


        # ==================================================
        # STAGE 5
        # SEMANTIC EVIDENCE VERIFIER
        # ==================================================

        self.semantic_verifier = (
            SemanticEvidenceVerifier()
        )


        self.final_k = final_k


        print(
            "\nRetrieval pipeline ready."
        )


    def search(
        self,
        query
    ):

        # ==================================================
        # STAGE 1
        # FAISS + BM25 + RRF
        # ==================================================

        candidates = (
            self.hybrid_retriever.search(
                query
            )
        )


        # ==================================================
        # STAGE 2
        # CROSS-ENCODER
        # ==================================================

        reranked = (
            self.reranker.rerank(

                query=query,

                candidates=candidates,

                top_k=self.final_k
            )
        )


        # ==================================================
        # STAGE 3
        # EVIDENCE GATE
        # ==================================================

        evidence = (
            self.evidence_gate.evaluate(

                query=query,

                results=reranked
            )
        )


        # ==================================================
        # STAGE 4
        # LEXICAL VERIFICATION
        # ==================================================

        if evidence["sufficient"]:

            verification = (
                self.evidence_verifier.verify(

                    query=query,

                    results=reranked
                )
            )

        else:

            verification = {

                "status":
                    "NOT_SUPPORTED",

                "reason":
                    "Evidence Gate rejected "
                    "the retrieved evidence.",

                "results": []
            }


        # ==================================================
        # STAGE 5
        # SEMANTIC VERIFICATION
        # ==================================================

        if (
            evidence["sufficient"]
            and verification["status"]
            in {
                "SUPPORTED",
                "PARTIALLY_SUPPORTED"
            }
        ):

            semantic_verification = (
                self.semantic_verifier.verify(

                    query=query,

                    results=reranked
                )
            )

        else:

            semantic_verification = {

                "verdict":
                    "NOT_SUPPORTED",

                "confidence":
                    1.0,

                "reason":
                    "Evidence did not pass "
                    "the previous verification stages."
            }


        # ==================================================
        # FINAL OUTPUT
        # ==================================================

        return {

            "query":
                query,

            "results":
                reranked,

            "evidence":
                evidence,

            "verification":
                verification,

            "semantic_verification":
                semantic_verification
        }


def print_results(
    output
):

    query = output[
        "query"
    ]

    results = output[
        "results"
    ]

    evidence = output[
        "evidence"
    ]

    verification = output[
        "verification"
    ]

    semantic = output[
        "semantic_verification"
    ]


    print("\n")

    print("=" * 80)

    print(
        f"QUERY: {query}"
    )

    print("=" * 80)


    # ======================================================
    # EVIDENCE GATE
    # ======================================================

    print(
        "\nEVIDENCE GATE"
    )


    print(
        "Status: "
        +
        (
            "PASS"
            if evidence["sufficient"]
            else "FAIL"
        )
    )


    if evidence[
        "best_score"
    ] is not None:

        print(
            f"Best Score: "
            f"{evidence['best_score']:.4f}"
        )

    else:

        print(
            "Best Score: None"
        )


    if evidence[
        "margin"
    ] is not None:

        print(
            f"Score Margin: "
            f"{evidence['margin']:.4f}"
        )

    else:

        print(
            "Score Margin: None"
        )


    print(
        f"Keyword Coverage: "
        f"{evidence['keyword_coverage']:.2f}"
    )


    print(
        f"Reason: "
        f"{evidence['reason']}"
    )


    # ======================================================
    # LEXICAL VERIFIER
    # ======================================================

    print(
        "\nLEXICAL EVIDENCE VERIFIER"
    )


    print(
        f"Status: "
        f"{verification['status']}"
    )


    print(
        f"Reason: "
        f"{verification['reason']}"
    )


    # ======================================================
    # SEMANTIC VERIFIER
    # ======================================================

    print(
        "\nSEMANTIC EVIDENCE VERIFIER"
    )


    print(
        f"Verdict: "
        f"{semantic['verdict']}"
    )


    print(
        f"Confidence: "
        f"{semantic['confidence']:.2f}"
    )


    print(
        f"Reason: "
        f"{semantic['reason']}"
    )


    # ======================================================
    # TOP EVIDENCE
    # ======================================================

    print(
        "\nTOP EVIDENCE"
    )


    for rank, result in enumerate(
        results,
        start=1
    ):

        chunk = result[
            "chunk"
        ]

        metadata = chunk[
            "metadata"
        ]


        print(
            f"\n[{rank}] "
            f"Reranker="
            f"{result['reranker_score']:.4f} "
            f"RRF="
            f"{result['rrf_score']:.6f}"
        )


        print(
            f"Vector Rank: "
            f"{result.get('vector_rank')}"
        )


        print(
            f"BM25 Rank: "
            f"{result.get('bm25_rank')}"
        )


        # --------------------------------------------------
        # Source
        # --------------------------------------------------

        print(
            f"Source: "
            f"{metadata.get('specification')}"
        )


        # --------------------------------------------------
        # Section
        # --------------------------------------------------

        section_number = metadata.get(
            "section_number"
        )

        parent_section = metadata.get(
            "parent_section"
        )

        section_title = metadata.get(
            "section_title"
        )


        if section_number:

            section_text = (
                f"{section_number} "
                f"{section_title}"
            )

        elif parent_section:

            section_text = (
                f"{parent_section} → "
                f"{section_title}"
            )

        else:

            section_text = (
                f"{section_title}"
            )


        print(
            f"Section: "
            f"{section_text}"
        )


        print(
            "Text:"
        )


        print(
            chunk["text"][:700]
        )


def main():

    pipeline = RetrievalPipeline(

        vector_k=20,

        bm25_k=20,

        hybrid_k=10,

        final_k=5
    )


    queries = [

        "What is the role of the AMF?",

        "What is the N11 interface?",

        "What is PDU Session Establishment?",

        "What is the role of the SMF?"
    ]


    for query in queries:

        output = pipeline.search(
            query
        )

        print_results(
            output
        )


if __name__ == "__main__":

    main()