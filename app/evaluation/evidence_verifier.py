import re


class EvidenceVerifier:

    def __init__(
        self,
        min_supporting_terms=0.5
    ):
        self.min_supporting_terms = (
            min_supporting_terms
        )

    # ======================================================
    # TOKENIZATION
    # ======================================================

    @staticmethod
    def tokenize(text):

        return set(
            re.findall(
                r"[a-zA-Z0-9]+(?:[-_/][a-zA-Z0-9]+)*",
                text.lower()
            )
        )

    # ======================================================
    # QUERY TERMS
    # ======================================================

    def important_terms(self, query):

        stop_words = {
            "what",
            "is",
            "the",
            "a",
            "an",
            "of",
            "to",
            "for",
            "in",
            "on",
            "and",
            "or",
            "does",
            "do",
            "how",
            "why",
            "role"
        }

        return {
            token
            for token in self.tokenize(query)
            if token not in stop_words
        }

    # ======================================================
    # VERIFY ONE EVIDENCE CHUNK
    # ======================================================

    def verify_chunk(
        self,
        query,
        text
    ):

        query_terms = self.important_terms(
            query
        )

        text_tokens = self.tokenize(
            text
        )

        if not query_terms:

            return {
                "label": "SUPPORTED",
                "score": 1.0,
                "matched_terms": []
            }

        matched_terms = (
            query_terms
            &
            text_tokens
        )

        score = (
            len(matched_terms)
            /
            len(query_terms)
        )

        if score >= self.min_supporting_terms:

            label = "SUPPORTED"

        elif score > 0:

            label = "PARTIALLY_SUPPORTED"

        else:

            label = "NOT_SUPPORTED"

        return {
            "label": label,
            "score": score,
            "matched_terms": sorted(
                matched_terms
            )
        }

    # ======================================================
    # VERIFY RETRIEVED RESULTS
    # ======================================================

    def verify(
        self,
        query,
        results
    ):

        if not results:

            return {
                "status": "NOT_SUPPORTED",
                "reason": "No evidence available.",
                "results": []
            }

        verified = []

        for result in results:

            text = result[
                "chunk"
            ]["text"]

            verification = (
                self.verify_chunk(
                    query,
                    text
                )
            )

            verified_result = result.copy()

            verified_result[
                "verification"
            ] = verification

            verified.append(
                verified_result
            )

        # --------------------------------------------------
        # Count supporting evidence
        # --------------------------------------------------

        supported = [
            item
            for item in verified
            if item["verification"]["label"]
            == "SUPPORTED"
        ]

        partially_supported = [
            item
            for item in verified
            if item["verification"]["label"]
            == "PARTIALLY_SUPPORTED"
        ]

        # --------------------------------------------------
        # Final decision
        # --------------------------------------------------

        if supported:

            status = "SUPPORTED"

            reason = (
                "At least one retrieved "
                "passage contains sufficient "
                "query-term evidence."
            )

        elif partially_supported:

            status = "PARTIALLY_SUPPORTED"

            reason = (
                "Retrieved passages contain "
                "some relevant terms but do "
                "not provide strong evidence."
            )

        else:

            status = "NOT_SUPPORTED"

            reason = (
                "Retrieved passages do not "
                "contain sufficient evidence "
                "for the query."
            )

        return {
            "status": status,
            "reason": reason,
            "results": verified
        }