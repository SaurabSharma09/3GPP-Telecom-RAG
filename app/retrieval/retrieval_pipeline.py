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

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker


# ==========================================================
# EVALUATION COMPONENTS
# ==========================================================

from app.evaluation.evidence_gate import EvidenceGate
from app.evaluation.evidence_verifier import EvidenceVerifier
from app.evaluation.semantic_verifier import SemanticEvidenceVerifier


# ==========================================================
# GENERATION COMPONENT
# ==========================================================

from app.generation.groq_generator import GroqGenerator


# ==========================================================
# RETRIEVAL + GENERATION PIPELINE
# ==========================================================


class RetrievalPipeline:
    """
    Canonical end-to-end telecom RAG pipeline.

    Flow:
        Query
        -> Hybrid retrieval
        -> Cross-encoder reranking
        -> Evidence gate
        -> Lexical verification
        -> Semantic verification
        -> Grounded Groq generation OR abstention
    """

    def __init__(self, vector_k=40, bm25_k=40, hybrid_k=20, final_k=15):

        print("=" * 80)
        print("INITIALIZING TELECOM RAG PIPELINE")
        print("=" * 80)

        # ==================================================
        # STAGE 1 - HYBRID RETRIEVAL
        # ==================================================

        print("\n[1/6] Loading hybrid retriever...")

        self.hybrid_retriever = HybridRetriever(
            vector_k=vector_k, bm25_k=bm25_k, final_k=hybrid_k
        )

        self.query_analyzer = self.hybrid_retriever.query_analyzer

        # ==================================================
        # STAGE 2 - CROSS-ENCODER
        # ==================================================

        print("\n[2/6] Loading cross-encoder reranker...")

        self.reranker = Reranker()

        # ==================================================
        # STAGE 3 - EVIDENCE GATE
        # ==================================================

        print("\n[3/6] Loading evidence gate...")

        self.evidence_gate = EvidenceGate(
            min_score=1.0, min_margin=0.5, min_keyword_coverage=0.5
        )

        # ==================================================
        # STAGE 4 - LEXICAL VERIFICATION
        # ==================================================

        print("\n[4/6] Loading lexical evidence verifier...")

        self.evidence_verifier = EvidenceVerifier(min_supporting_terms=0.5)

        # ==================================================
        # STAGE 5 - SEMANTIC VERIFICATION
        # ==================================================

        print("\n[5/6] Loading semantic evidence verifier...")

        self.semantic_verifier = SemanticEvidenceVerifier()

        # ==================================================
        # STAGE 6 - GROQ GENERATOR
        # ==================================================

        print("\n[6/6] Loading Groq generator...")

        self.generator = GroqGenerator()

        self.final_k = final_k

        print("\n" + "=" * 80)
        print("TELECOM RAG PIPELINE READY")
        print("=" * 80)

    # ======================================================
    # SOURCE SUMMARY
    # ======================================================

    @staticmethod
    def build_sources(results):
        """
        Return a deduplicated source list for UI/reporting.
        """

        sources = []
        seen = set()

        for result in results:

            chunk = result.get("chunk", {})
            metadata = chunk.get("metadata", {})

            specification = metadata.get("specification")

            section_number = metadata.get("section_number")

            section_title = metadata.get("section_title")

            source_file = metadata.get("source_file")

            section_path = metadata.get("section_path", [])

            key = (specification, section_number, section_title)

            if key in seen:
                continue

            seen.add(key)

            sources.append(
                {
                    "specification": specification,
                    "section_number": section_number,
                    "section_title": section_title,
                    "source_file": source_file,
                    "section_path": section_path,
                }
            )

        return sources

    # ======================================================
    # GROUNDING SUMMARY
    # ======================================================

    @staticmethod
    def build_grounding_summary(final_status, results):

        sources = RetrievalPipeline.build_sources(results)

        if final_status == "GROUNDED":

            return {
                "status": "SUPPORTED",
                "evidence_chunks": len(results),
                "source_count": len(sources),
                "message": (
                    "Answer is supported by retrieved " "and verified 3GPP evidence."
                ),
            }

        if final_status == "PARTIALLY_SUPPORTED":

            return {
                "status": "PARTIALLY_SUPPORTED",
                "evidence_chunks": len(results),
                "source_count": len(sources),
                "message": (
                    "Only part of the requested information "
                    "is supported by the retrieved 3GPP evidence."
                ),
            }

        return {
            "status": "NOT_SUPPORTED",
            "evidence_chunks": len(results),
            "source_count": len(sources),
            "message": (
                "The available 3GPP evidence is insufficient " "to answer the question."
            ),
        }

    # ======================================================
    # SEARCH
    # ======================================================

    def search(self, query):

        # ==================================================
        # STAGE 0 - QUERY ANALYSIS
        # ==================================================

        query_analysis = self.query_analyzer.analyze(query)

        # ==================================================
        # STAGE 1 - HYBRID RETRIEVAL
        # ==================================================

        print("\n[STAGE 1] Hybrid retrieval...")

        candidates = self.hybrid_retriever.search(query)

        print(f"Candidates retrieved: {len(candidates)}")

        # ==================================================
        # STAGE 2 - CROSS-ENCODER RERANKING
        # ==================================================

        print("\n[STAGE 2] Cross-encoder reranking...")

        reranked = self.reranker.rerank(
            query=query, candidates=candidates, top_k=self.final_k
        )

        print(f"Candidates after reranking: " f"{len(reranked)}")

        # ==================================================
        # STAGE 3 - EVIDENCE GATE
        # ==================================================

        print("\n[STAGE 3] Evidence gate...")

        evidence = self.evidence_gate.evaluate(query=query, results=reranked)

        print("Evidence Gate: " + ("PASS" if evidence["sufficient"] else "FAIL"))

        # ==================================================
        # STAGE 4 - LEXICAL VERIFICATION
        # ==================================================

        print("\n[STAGE 4] Lexical evidence verification...")

        if evidence["sufficient"]:

            verification = self.evidence_verifier.verify(query=query, results=reranked)

        else:

            verification = {
                "status": "NOT_SUPPORTED",
                "reason": ("Evidence Gate rejected " "the retrieved evidence."),
                "results": [],
            }

        print("Evidence Verifier: " f"{verification['status']}")

        # ==================================================
        # STAGE 5 - SEMANTIC VERIFICATION
        # ==================================================

        print("\n[STAGE 5] Semantic verification...")

        if evidence["sufficient"] and verification["status"] in {
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
        }:

            semantic_verification = self.semantic_verifier.verify(
                query=query, results=reranked
            )

        else:

            semantic_verification = {
                "verdict": "NOT_SUPPORTED",
                "confidence": 1.0,
                "reason": (
                    "Evidence did not pass " "the previous verification stages."
                ),
            }

        print("Semantic Verifier: " f"{semantic_verification['verdict']}")

        # ==================================================
        # STAGE 6 - GROUNDED GENERATION
        # ==================================================

        print("\n[STAGE 6] Generating grounded answer...")

        # Default to abstention.
        answer = (
            "The retrieved 3GPP evidence is " "insufficient to answer this question."
        )

        final_status = "NOT_SUPPORTED"

        # Groq is called ONLY after all evidence checks pass.
        if (
            evidence["sufficient"]
            and verification["status"] == "SUPPORTED"
            and semantic_verification["verdict"] == "SUPPORTED"
        ):

            answer = self.generator.generate(query=query, results=reranked)

            final_status = "GROUNDED"

        elif semantic_verification["verdict"] == "PARTIALLY_SUPPORTED":

            answer = (
                "The retrieved 3GPP evidence only partially "
                "supports this question, so no unsupported "
                "answer was generated."
            )

            final_status = "PARTIALLY_SUPPORTED"

        # ==================================================
        # SOURCES + GROUNDING
        # ==================================================

        sources = self.build_sources(reranked)

        grounding = self.build_grounding_summary(
            final_status=final_status, results=reranked
        )

        # ==================================================
        # FINAL OUTPUT
        # ==================================================

        return {
            "query": query,
            "query_analysis": query_analysis,
            "answer": answer,
            "final_status": final_status,
            "sources": sources,
            "grounding": grounding,
            "results": reranked,
            "evidence": evidence,
            "verification": verification,
            "semantic_verification": semantic_verification,
        }


# ==========================================================
# PRINT RESULTS
# ==========================================================


def print_results(output):

    query = output["query"]
    answer = output["answer"]
    results = output["results"]

    evidence = output["evidence"]
    verification = output["verification"]
    semantic = output["semantic_verification"]

    sources = output.get("sources", [])

    grounding = output.get("grounding", {})

    # ======================================================
    # QUERY
    # ======================================================

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    # ======================================================
    # ANSWER
    # ======================================================

    print("\nANSWER")
    print("-" * 80)
    print(answer)

    # ======================================================
    # FINAL STATUS
    # ======================================================

    print("\nFINAL STATUS")
    print("-" * 80)
    print(
        f"{output['final_status']} | "
        f"Grounding={grounding.get('status')} | "
        f"Evidence={grounding.get('evidence_chunks', 0)} | "
        f"Sources={grounding.get('source_count', 0)}"
    )

    # ======================================================
    # EVIDENCE GATE
    # ======================================================

    print("\nEVIDENCE GATE")
    print("-" * 80)

    print("Status: " + ("PASS" if evidence["sufficient"] else "FAIL"))

    print(f"Best Score: " f"{evidence.get('best_score')}")

    print(f"Margin: " f"{evidence.get('margin')}")

    print(f"Keyword Coverage: " f"{evidence.get('keyword_coverage', 0):.2f}")

    print(f"Reason: " f"{evidence.get('reason', '')}")

    # ======================================================
    # VERIFIERS
    # ======================================================

    print("\nVERIFICATION")
    print("-" * 80)

    print(f"Lexical: " f"{verification.get('status')}")

    print(f"Semantic: " f"{semantic.get('verdict')}")

    print(f"Semantic Confidence: " f"{semantic.get('confidence', 0):.2f}")

    # ======================================================
    # SOURCES
    # ======================================================

    print("\nSOURCES")
    print("-" * 80)

    if not sources:
        print("No verified sources.")

    for index, source in enumerate(sources, start=1):

        print(
            f"[{index}] "
            f"{source.get('specification')} | "
            f"{source.get('section_number')} | "
            f"{source.get('section_title')}"
        )

    # ======================================================
    # TOP EVIDENCE
    # ======================================================

    print("\nTOP EVIDENCE")
    print("-" * 80)

    for rank, result in enumerate(results, start=1):

        chunk = result["chunk"]
        metadata = chunk["metadata"]

        print(
            f"\n[{rank}] "
            f"Combined="
            f"{result.get('combined_rerank_score', 0.0):.4f} "
            f"CrossEncoder="
            f"{result.get('reranker_score', 0.0):.4f} "
            f"RRF="
            f"{result.get('rrf_score', 0.0):.6f} "
            f"Canonical="
            f"{result.get('canonical_concept_score', 0.0):.2f}"
        )

        print(f"Source: " f"{metadata.get('specification')}")

        print(
            f"Section: "
            f"{metadata.get('section_number')} "
            f"{metadata.get('section_title')}"
        )

        print(f"Chunk type: " f"{metadata.get('chunk_type')}")

        print("Text:")

        print(chunk["text"][:700])


# ==========================================================
# TEST
# ==========================================================


def main():

    pipeline = RetrievalPipeline(vector_k=40, bm25_k=40, hybrid_k=20, final_k=15)

    queries = [
        "What is the role of the AMF?",
        "What is the role of the SMF?",
        "What is the role of the UPF?",
        "What is the N4 interface?",
        "What is PDU Session Establishment?",
        "How does UE registration work in the 5G system?",
        "What is the difference between AMF and SMF?",
        "What is the capital of France?",
    ]

    for query in queries:

        output = pipeline.search(query)

        print_results(output)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
