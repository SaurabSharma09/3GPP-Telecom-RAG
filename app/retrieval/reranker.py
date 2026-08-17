from sentence_transformers import CrossEncoder


# ============================================================================
# MODEL
# ============================================================================

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================================
# RERANKER
# ============================================================================


class Reranker:
    """
    Second-stage reranker for the telecom RAG system.

    Combines:
        60% CrossEncoder semantic relevance
        40% hybrid retrieval relevance

    The hybrid score preserves telecom-specific signals such as:
        - entity relevance
        - concept relevance
        - section relevance
        - canonical section relevance
        - interface relevance
    """

    def __init__(self):

        print("Loading cross-encoder reranker...")

        self.model = CrossEncoder(MODEL_NAME)

        print("Reranker ready.")

    # ========================================================================
    # SCORE NORMALIZATION
    # ========================================================================

    @staticmethod
    def normalize_scores(scores):
        """
        Min-max normalize scores to [0, 1].

        If all scores are identical, return 0.5 for every item instead
        of creating an arbitrary ranking from identical values.
        """

        if not scores:
            return []

        minimum = min(scores)

        maximum = max(scores)

        if maximum == minimum:

            return [0.5 for _ in scores]

        return [(score - minimum) / (maximum - minimum) for score in scores]

    # ========================================================================
    # RERANK
    # ========================================================================

    def rerank(self, query, candidates, top_k=5):
        """
        Rerank hybrid-retrieval candidates.

        Final score:

            60% CrossEncoder
            40% Hybrid retrieval

        In addition, a strong canonical 3GPP section receives a small
        deterministic bonus. This prevents a generic cross-encoder from
        pushing a canonical standard clause below a specialized clause
        that merely happens to contain similar language.
        """

        if not candidates:
            return []

        # --------------------------------------------------------------------
        # Build CrossEncoder query-document pairs
        # --------------------------------------------------------------------

        pairs = []

        for candidate in candidates:

            chunk = candidate.get("chunk", {})

            text = chunk.get("text", "")

            pairs.append((query, text))

        # --------------------------------------------------------------------
        # CrossEncoder scores
        # --------------------------------------------------------------------

        scores = self.model.predict(pairs)

        reranker_scores = [float(score) for score in scores]

        # --------------------------------------------------------------------
        # Hybrid retrieval scores
        # --------------------------------------------------------------------

        hybrid_scores = [
            float(
                candidate.get("final_retrieval_score", candidate.get("rrf_score", 0.0))
            )
            for candidate in candidates
        ]

        normalized_reranker = self.normalize_scores(reranker_scores)

        normalized_hybrid = self.normalize_scores(hybrid_scores)

        # --------------------------------------------------------------------
        # Final weighting
        # --------------------------------------------------------------------

        CROSS_ENCODER_WEIGHT = 0.60
        HYBRID_WEIGHT = 0.40

        # Small deterministic protection for canonical telecom sections.
        CANONICAL_SECTION_BONUS = 0.15

        reranked = []

        for candidate, reranker_score, normalized_ce, normalized_hybrid_score in zip(
            candidates, reranker_scores, normalized_reranker, normalized_hybrid
        ):

            result = candidate.copy()

            # --------------------------------------------------------------
            # Raw CrossEncoder score
            # --------------------------------------------------------------

            result["reranker_score"] = reranker_score

            # --------------------------------------------------------------
            # Normalized scores
            # --------------------------------------------------------------

            result["normalized_reranker_score"] = normalized_ce

            result["normalized_hybrid_score"] = normalized_hybrid_score

            # --------------------------------------------------------------
            # Base combined score
            # --------------------------------------------------------------

            combined_score = (
                CROSS_ENCODER_WEIGHT * normalized_ce
                + HYBRID_WEIGHT * normalized_hybrid_score
            )

            # --------------------------------------------------------------
            # Canonical 3GPP safeguard
            # --------------------------------------------------------------

            canonical_score = float(result.get("canonical_concept_score", 0.0))

            if canonical_score >= 1.0:

                combined_score += CANONICAL_SECTION_BONUS

                result["canonical_section_bonus"] = CANONICAL_SECTION_BONUS

            else:

                result["canonical_section_bonus"] = 0.0

            # --------------------------------------------------------------
            # Save final score
            # --------------------------------------------------------------

            result["combined_rerank_score"] = combined_score

            reranked.append(result)

        # --------------------------------------------------------------------
        # Final ordering
        # --------------------------------------------------------------------

        reranked.sort(key=lambda item: item["combined_rerank_score"], reverse=True)

        return reranked[:top_k]


# ============================================================================
# TEST
# ============================================================================


def main():

    reranker = Reranker()

    query = "What is the N11 interface?"

    candidates = [
        {
            "final_retrieval_score": 0.80,
            "canonical_concept_score": 0.0,
            "chunk": {
                "text": "The N11 interface "
                "supports communication "
                "between network functions."
            },
        },
        {
            "final_retrieval_score": 0.30,
            "canonical_concept_score": 0.0,
            "chunk": {"text": "The N2 interface " "connects the 5G-AN " "and AMF."},
        },
    ]

    results = reranker.rerank(query, candidates, top_k=2)

    print("\nReranking test:")

    for index, result in enumerate(results, start=1):

        print(f"\n[{index}]")

        print(f"CrossEncoder: " f"{result['reranker_score']:.4f}")

        print(f"Hybrid: " f"{result['normalized_hybrid_score']:.4f}")

        print(f"Canonical Bonus: " f"{result['canonical_section_bonus']:.4f}")

        print(f"Combined: " f"{result['combined_rerank_score']:.4f}")

        print(result["chunk"]["text"])

        print("-" * 60)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
