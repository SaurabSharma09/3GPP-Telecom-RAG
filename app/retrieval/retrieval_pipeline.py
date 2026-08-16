from pathlib import Path
import sys


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# RETRIEVAL COMPONENTS
# ==========================================================

from app.retrieval.hybrid_retriever import (
    HybridRetriever
)

from app.retrieval.reranker import (
    Reranker
)


# ==========================================================
# EVALUATION COMPONENTS
# ==========================================================

from app.evaluation.evidence_gate import (
    EvidenceGate
)

from app.evaluation.evidence_verifier import (
    EvidenceVerifier
)


# ==========================================================
# GENERATION COMPONENT
# ==========================================================

from app.generation.groq_generator import (
    GroqGenerator
)


# ==========================================================
# RETRIEVAL + GENERATION PIPELINE
# ==========================================================

class RetrievalPipeline:

    def __init__(
        self,
        vector_k=20,
        bm25_k=20,
        hybrid_k=10,
        final_k=5
    ):

        print("=" * 80)

        print(
            "INITIALIZING TELECOM RAG PIPELINE"
        )

        print("=" * 80)

        # ==================================================
        # STAGE 1
        # HYBRID RETRIEVAL
        # ==================================================

        print(
            "\n[1/5] Loading hybrid retriever..."
        )

        self.hybrid_retriever = (
            HybridRetriever(
                vector_k=vector_k,
                bm25_k=bm25_k,
                final_k=hybrid_k
            )
        )

        # ==================================================
        # STAGE 2
        # CROSS-ENCODER RERANKER
        # ==================================================

        print(
            "\n[2/5] Loading cross-encoder reranker..."
        )

        self.reranker = Reranker()

        # ==================================================
        # STAGE 3
        # EVIDENCE GATE
        # ==================================================

        print(
            "\n[3/5] Loading evidence gate..."
        )

        self.evidence_gate = EvidenceGate(

            min_score=1.0,

            min_margin=0.5,

            min_keyword_coverage=0.5
        )

        # ==================================================
        # STAGE 4
        # EVIDENCE VERIFIER
        # ==================================================

        print(
            "\n[4/5] Loading evidence verifier..."
        )

        self.evidence_verifier = (
            EvidenceVerifier(
                min_supporting_terms=0.5
            )
        )

        # ==================================================
        # STAGE 5
        # GROQ GENERATOR
        # ==================================================

        print(
            "\n[5/5] Loading Groq generator..."
        )

        self.generator = GroqGenerator()

        self.final_k = final_k

        print("\n" + "=" * 80)

        print(
            "TELECOM RAG PIPELINE READY"
        )

        print("=" * 80)


    # ======================================================
    # SEARCH
    # ======================================================

    def search(
        self,
        query
    ):

        # ==================================================
        # STAGE 1
        # HYBRID RETRIEVAL
        # ==================================================

        print(
            "\n[STAGE 1] Hybrid retrieval..."
        )

        candidates = (
            self.hybrid_retriever.search(
                query
            )
        )

        print(
            f"Candidates retrieved: "
            f"{len(candidates)}"
        )

        # ==================================================
        # STAGE 2
        # CROSS-ENCODER RERANKING
        # ==================================================

        print(
            "\n[STAGE 2] Cross-encoder reranking..."
        )

        reranked = (
            self.reranker.rerank(

                query=query,

                candidates=candidates,

                top_k=self.final_k
            )
        )

        print(
            f"Candidates after reranking: "
            f"{len(reranked)}"
        )

        # ==================================================
        # STAGE 3
        # EVIDENCE GATE
        # ==================================================

        print(
            "\n[STAGE 3] Evidence gate..."
        )

        evidence = (
            self.evidence_gate.evaluate(

                query=query,

                results=reranked
            )
        )

        print(
            "Evidence Gate: "
            +
            (
                "PASS"
                if evidence["sufficient"]
                else "FAIL"
            )
        )

        # ==================================================
        # STAGE 4
        # EVIDENCE VERIFICATION
        # ==================================================

        print(
            "\n[STAGE 4] Evidence verification..."
        )

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
                    (
                        "Evidence Gate rejected "
                        "the retrieved evidence."
                    ),

                "results":
                    []
            }

        print(
            "Evidence Verifier: "
            +
            verification["status"]
        )

        # ==================================================
        # STAGE 5
        # GROQ GENERATION
        # ==================================================

        print(
            "\n[STAGE 5] Generating grounded answer..."
        )

        answer = None

        # --------------------------------------------------
        # IMPORTANT:
        #
        # Groq is called ONLY when both:
        #
        # Evidence Gate      → PASS
        # Evidence Verifier  → SUPPORTED
        #
        # This prevents the LLM from filling missing
        # evidence with its own knowledge.
        # --------------------------------------------------

        if (

            evidence["sufficient"]

            and

            verification["status"]
            == "SUPPORTED"

        ):

            answer = (
                self.generator.generate(

                    query=query,

                    results=reranked
                )
            )

        else:

            answer = (
                "The retrieved 3GPP evidence is "
                "insufficient to answer this question."
            )

        # ==================================================
        # FINAL OUTPUT
        # ==================================================

        return {

            "query":
                query,

            "answer":
                answer,

            "results":
                reranked,

            "evidence":
                evidence,

            "verification":
                verification
        }


# ==========================================================
# PRINT RESULTS
# ==========================================================

def print_results(
    output
):

    query = output[
        "query"
    ]

    answer = output[
        "answer"
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


    # ======================================================
    # QUERY
    # ======================================================

    print("\n")

    print("=" * 80)

    print(
        f"QUERY: {query}"
    )

    print("=" * 80)


    # ======================================================
    # GROUNDED ANSWER
    # ======================================================

    print("\n" + "=" * 80)

    print(
        "GROUNDED ANSWER"
    )

    print("=" * 80)

    print(
        answer
    )


    # ======================================================
    # EVIDENCE GATE
    # ======================================================

    print("\n" + "=" * 80)

    print(
        "EVIDENCE GATE"
    )

    print("=" * 80)

    print(

        "Status: "

        +

        (
            "PASS"
            if evidence["sufficient"]
            else "FAIL"
        )
    )

    if evidence["best_score"] is not None:

        print(
            f"Best Score: "
            f"{evidence['best_score']:.4f}"
        )

    else:

        print(
            "Best Score: None"
        )


    if evidence["margin"] is not None:

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
    # EVIDENCE VERIFIER
    # ======================================================

    print("\n" + "=" * 80)

    print(
        "EVIDENCE VERIFIER"
    )

    print("=" * 80)

    print(
        f"Status: "
        f"{verification['status']}"
    )

    print(
        f"Reason: "
        f"{verification['reason']}"
    )


    # ======================================================
    # VERIFICATION DETAILS
    # ======================================================

    if verification["results"]:

        print(
            "\nVERIFICATION DETAILS"
        )

        for index, item in enumerate(

            verification["results"],

            start=1
        ):

            check = item[
                "verification"
            ]

            print(

                f"\n[{index}] "

                f"{check['label']} "

                f"score="

                f"{check['score']:.2f}"
            )

            print(

                f"Matched terms: "

                f"{check['matched_terms']}"
            )


    # ======================================================
    # TOP EVIDENCE
    # ======================================================

    print("\n" + "=" * 80)

    print(
        "TOP EVIDENCE"
    )

    print("=" * 80)


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


        # --------------------------------------------------
        # RANKING
        # --------------------------------------------------

        print(

            f"\n[{rank}] "

            f"Reranker="

            f"{result['reranker_score']:.4f}"
        )


        print(

            f"RRF="

            f"{result.get('rrf_score', 0):.6f}"
        )


        print(

            f"Entity="

            f"{result.get('entity_score', 0):.3f}"
        )


        print(

            f"Coverage="

            f"{result.get('entity_coverage', 0):.2f}"
        )


        print(

            f"Intent="

            f"{result.get('intent_score', 0):.3f}"
        )


        print(

            f"Concept="

            f"{result.get('concept_score', 0):.2f}"
        )


        print(

            f"Section="

            f"{result.get('section_score', 0):.2f}"
        )


        print(

            f"Interface="

            f"{result.get('interface_score', 0):.2f}"
        )


        print(

            f"ExactInterface="

            f"{result.get('exact_interface_score', 0):.2f}"
        )


        print(

            f"Vector Rank="

            f"{result.get('vector_rank')}"
        )


        print(

            f"BM25 Rank="

            f"{result.get('bm25_rank')}"
        )


        # --------------------------------------------------
        # SOURCE
        # --------------------------------------------------

        print(

            f"Source: "

            f"{metadata.get('specification')}"
        )


        # --------------------------------------------------
        # SECTION
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


        # --------------------------------------------------
        # EVIDENCE TEXT
        # --------------------------------------------------

        print(
            "Text:"
        )

        print(
            chunk["text"][:700]
        )


# ==========================================================
# TEST
# ==========================================================

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

        "How does PDU Session Establishment work?",

        "What is the role of the SMF?",

        "What is the difference between AMF and SMF?",

        "How does the AMF connect to the SMF?"
    ]


    for query in queries:

        output = pipeline.search(
            query
        )

        print_results(
            output
        )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()