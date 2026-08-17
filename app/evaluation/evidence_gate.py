import re


class EvidenceGate:

    def __init__(self, min_score=0.45, min_margin=0.05, min_keyword_coverage=0.5):

        self.min_score = min_score
        self.min_margin = min_margin
        self.min_keyword_coverage = min_keyword_coverage

    # ==========================================================
    # QUERY TYPE HELPERS
    # ==========================================================

    def _is_interface_query(self, query):

        query_lower = query.lower().strip()

        interface_words = [
            "interface",
            "reference point",
            "connect",
            "connection",
            "interact",
            "interaction",
        ]

        has_interface_word = any(word in query_lower for word in interface_words)

        interface_entity = re.search(r"\bN\d+[A-Za-z]?\b", query, re.IGNORECASE)

        return has_interface_word or interface_entity is not None

    # ==========================================================
    # TECHNICAL INTERFACE
    # ==========================================================

    def _extract_interface(self, query):

        match = re.search(r"\b(N\d+[A-Za-z]?)\b", query, re.IGNORECASE)

        if match:
            return match.group(1).upper()

        return None

    def _check_interface_evidence(self, query, results):

        interface = self._extract_interface(query)

        if not interface:
            return {"supported": False, "score": 0.0, "reason": None}

        best_score = 0.0
        best_result = None

        for result in results:

            chunk = result.get("chunk", {})

            text = chunk.get("text", "")

            text_upper = text.upper()

            exact_text_match = bool(
                re.search(rf"\b{re.escape(interface)}\b", text_upper)
            )

            if not exact_text_match:
                continue

            interface_score = float(result.get("interface_score", 0.0))

            exact_interface_score = float(result.get("exact_interface_score", 0.0))

            entity_score = float(result.get("entity_score", 0.0))

            entity_coverage = float(
                result.get("entity_coverage", result.get("coverage", 0.0))
            )

            technical_score = max(exact_interface_score, interface_score)

            if exact_interface_score >= 0.25:

                candidate_score = 1.0 + technical_score + entity_score + entity_coverage

            elif interface_score >= 0.40 and entity_coverage >= 0.5:

                candidate_score = 0.8 + interface_score + entity_coverage

            else:

                candidate_score = technical_score

            if candidate_score > best_score:

                best_score = candidate_score

                best_result = result

        if best_result is None:

            return {
                "supported": False,
                "score": 0.0,
                "reason": "No direct interface evidence found.",
            }

        best_exact = float(best_result.get("exact_interface_score", 0.0))

        best_interface = float(best_result.get("interface_score", 0.0))

        best_entity = float(best_result.get("entity_score", 0.0))

        best_coverage = float(
            best_result.get("entity_coverage", best_result.get("coverage", 0.0))
        )

        # ------------------------------------------------------
        # Exact interface evidence
        # ------------------------------------------------------

        if best_exact >= 0.25 and best_coverage >= 0.5:

            return {
                "supported": True,
                "score": best_score,
                "reason": (
                    f"Direct {interface} interface "
                    "evidence found in retrieved "
                    "3GPP text."
                ),
            }

        # ------------------------------------------------------
        # Strong interface evidence
        # ------------------------------------------------------

        if best_interface >= 0.40 and best_entity >= 0.4 and best_coverage >= 0.5:

            return {
                "supported": True,
                "score": best_score,
                "reason": (
                    f"Strong {interface} interface "
                    "evidence found in retrieved "
                    "3GPP text."
                ),
            }

        return {
            "supported": False,
            "score": best_score,
            "reason": (
                f"{interface} was found, but "
                "interface evidence was not strong "
                "enough."
            ),
        }

    # ==========================================================
    # GET FINAL RERANK SCORE
    # ==========================================================

    @staticmethod
    def _get_final_score(result):

        # Prefer the score actually used by the final reranker.
        if "combined_rerank_score" in result:

            return float(result.get("combined_rerank_score", 0.0))

        # Backward compatibility.
        return float(result.get("reranker_score", 0.0))

    # ==========================================================
    # MAIN EVALUATION
    # ==========================================================

    def evaluate(self, query, results):

        if not results:

            return {
                "sufficient": False,
                "best_score": None,
                "margin": None,
                "keyword_coverage": 0.0,
                "reason": "No retrieval results found.",
            }

        # ======================================================
        # FINAL RERANK SCORES
        # ======================================================

        scores = [self._get_final_score(result) for result in results]

        ranked_scores = sorted(scores, reverse=True)

        best_score = ranked_scores[0]

        margin = (
            ranked_scores[0] - ranked_scores[1]
            if len(ranked_scores) > 1
            else ranked_scores[0]
        )

        # ======================================================
        # KEYWORD COVERAGE
        # ======================================================

        query_words = [
            word.lower()
            for word in re.findall(r"\b[a-zA-Z0-9]+\b", query)
            if len(word) > 2
        ]

        # For technical questions, use the actual evidence text.
        # Section metadata is intentionally ignored here so terms from
        # Cover/document metadata cannot manufacture keyword coverage.
        matched_words = set()

        for result in results:

            chunk = result.get("chunk", {})

            metadata = chunk.get("metadata", {})

            # Exclude cover chunks from normal technical evidence.
            if metadata.get("section_type") == "cover":
                continue

            text = chunk.get("text", "").lower()

            for word in query_words:

                if word in text:

                    matched_words.add(word)

        if query_words:

            keyword_coverage = len(matched_words) / len(set(query_words))

        else:

            keyword_coverage = 0.0

        # ======================================================
        # INTERFACE OVERRIDE
        # ======================================================

        if self._is_interface_query(query):

            interface_check = self._check_interface_evidence(
                query=query, results=results
            )

            if interface_check["supported"]:

                return {
                    "sufficient": True,
                    "best_score": best_score,
                    "margin": margin,
                    "keyword_coverage": keyword_coverage,
                    "reason": interface_check["reason"],
                }

        # ======================================================
        # NORMAL EVIDENCE RULES
        # ======================================================

        # ------------------------------------------------------
        # Rule 1: strong final rerank score + coverage
        # ------------------------------------------------------

        if (
            best_score >= self.min_score
            and keyword_coverage >= self.min_keyword_coverage
        ):

            return {
                "sufficient": True,
                "best_score": best_score,
                "margin": margin,
                "keyword_coverage": keyword_coverage,
                "reason": "Evidence passed all checks.",
            }

        # ------------------------------------------------------
        # Rule 2: strong telecom semantic signal
        # ------------------------------------------------------

        top_result = results[0]

        entity_score = float(top_result.get("entity_score", 0.0))

        entity_coverage = float(
            top_result.get("entity_coverage", top_result.get("coverage", 0.0))
        )

        intent_score = float(top_result.get("intent_score", 0.0))

        concept_score = float(top_result.get("concept_score", 0.0))

        canonical_score = float(top_result.get("canonical_concept_score", 0.0))

        strong_semantic_evidence = (
            entity_score >= 0.5
            and entity_coverage >= 0.5
            and (intent_score >= 0.7 or concept_score >= 0.7 or canonical_score >= 1.0)
        )

        if strong_semantic_evidence and best_score >= 0.25:

            return {
                "sufficient": True,
                "best_score": best_score,
                "margin": margin,
                "keyword_coverage": keyword_coverage,
                "reason": (
                    "Evidence accepted because "
                    "strong telecom entity/concept/"
                    "intent/canonical signals support "
                    "the result."
                ),
            }

        # ------------------------------------------------------
        # Reject
        # ------------------------------------------------------

        if best_score < self.min_score:

            reason = "Best final rerank score is too low."

        elif keyword_coverage < self.min_keyword_coverage:

            reason = "Keyword coverage is too low."

        elif margin < self.min_margin:

            reason = "Score margin between top results " "is too small."

        else:

            reason = "Evidence did not satisfy " "minimum requirements."

        return {
            "sufficient": False,
            "best_score": best_score,
            "margin": margin,
            "keyword_coverage": keyword_coverage,
            "reason": reason,
        }
