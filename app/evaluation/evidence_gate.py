import re


class EvidenceGate:

    def __init__(
        self,
        min_score=1.0,
        min_margin=0.5,
        min_keyword_coverage=0.5
    ):

        self.min_score = min_score
        self.min_margin = min_margin
        self.min_keyword_coverage = min_keyword_coverage


    # ==========================================================
    # QUERY TYPE HELPERS
    # ==========================================================

    def _is_interface_query(self, query):

        query_lower = query.lower()

        interface_words = [
            "interface",
            "reference point",
            "connect",
            "connection",
            "interact",
            "interaction"
        ]

        has_interface_word = any(
            word in query_lower
            for word in interface_words
        )

        # Examples:
        # N11
        # N2
        # N3
        # N4
        # N6
        # N14
        interface_entity = re.search(
            r"\bN\d+[A-Za-z]?\b",
            query,
            re.IGNORECASE
        )

        return (
            has_interface_word
            or interface_entity is not None
        )


    # ==========================================================
    # EXTRACT TECHNICAL INTERFACE
    # ==========================================================

    def _extract_interface(self, query):

        match = re.search(
            r"\b(N\d+[A-Za-z]?)\b",
            query,
            re.IGNORECASE
        )

        if match:
            return match.group(1).upper()

        return None


    # ==========================================================
    # CHECK DIRECT INTERFACE EVIDENCE
    # ==========================================================

    def _check_interface_evidence(
        self,
        query,
        results
    ):

        interface = self._extract_interface(
            query
        )

        if not interface:
            return {
                "supported": False,
                "score": 0.0,
                "reason": None
            }

        best_score = 0.0
        best_result = None

        for result in results:

            chunk = result.get(
                "chunk",
                {}
            )

            text = chunk.get(
                "text",
                ""
            )

            text_upper = text.upper()

            # ----------------------------------------------
            # Exact textual occurrence
            # ----------------------------------------------

            exact_text_match = bool(
                re.search(
                    rf"\b{re.escape(interface)}\b",
                    text_upper
                )
            )

            if not exact_text_match:
                continue

            # ----------------------------------------------
            # Retrieval features
            # ----------------------------------------------

            interface_score = float(
                result.get(
                    "interface_score",
                    0.0
                )
            )

            exact_interface_score = float(
                result.get(
                    "exact_interface_score",
                    0.0
                )
            )

            entity_score = float(
                result.get(
                    "entity_score",
                    0.0
                )
            )

            entity_coverage = float(
                result.get(
                    "entity_coverage",
                    result.get(
                        "coverage",
                        0.0
                    )
                )
            )

            reranker_score = float(
                result.get(
                    "reranker_score",
                    0.0
                )
            )

            # ----------------------------------------------
            # Calculate technical evidence score
            # ----------------------------------------------

            technical_score = max(
                exact_interface_score,
                interface_score
            )

            # Strong direct evidence
            if exact_interface_score >= 0.25:

                candidate_score = (
                    1.0
                    + technical_score
                    + entity_score
                    + entity_coverage
                )

            # Good interface evidence
            elif (
                interface_score >= 0.40
                and entity_coverage >= 0.5
            ):

                candidate_score = (
                    0.8
                    + interface_score
                    + entity_coverage
                )

            else:

                candidate_score = (
                    technical_score
                )

            if candidate_score > best_score:

                best_score = candidate_score

                best_result = result


        if best_result is None:

            return {
                "supported": False,
                "score": 0.0,
                "reason":
                    "No direct interface evidence found."
            }


        # ======================================================
        # INTERFACE EVIDENCE ACCEPTANCE
        # ======================================================

        best_exact = float(
            best_result.get(
                "exact_interface_score",
                0.0
            )
        )

        best_interface = float(
            best_result.get(
                "interface_score",
                0.0
            )
        )

        best_entity = float(
            best_result.get(
                "entity_score",
                0.0
            )
        )

        best_coverage = float(
            best_result.get(
                "entity_coverage",
                best_result.get(
                    "coverage",
                    0.0
                )
            )
        )


        # ------------------------------------------------------
        # Rule 1:
        # Exact interface evidence
        # ------------------------------------------------------

        if (
            best_exact >= 0.25
            and best_coverage >= 0.5
        ):

            return {

                "supported": True,

                "score": best_score,

                "reason":
                    (
                        f"Direct {interface} interface "
                        "evidence found in retrieved "
                        "3GPP text."
                    )
            }


        # ------------------------------------------------------
        # Rule 2:
        # Strong interface evidence
        # ------------------------------------------------------

        if (
            best_interface >= 0.40
            and best_entity >= 0.4
            and best_coverage >= 0.5
        ):

            return {

                "supported": True,

                "score": best_score,

                "reason":
                    (
                        f"Strong {interface} interface "
                        "evidence found despite low "
                        "cross-encoder score."
                    )
            }


        return {

            "supported": False,

            "score": best_score,

            "reason":
                (
                    f"{interface} was found, but "
                    "interface evidence was not strong "
                    "enough."
                )
        }


    # ==========================================================
    # MAIN EVALUATION
    # ==========================================================

    def evaluate(
        self,
        query,
        results
    ):

        if not results:

            return {

                "sufficient": False,

                "best_score": None,

                "margin": None,

                "keyword_coverage": 0.0,

                "reason":
                    "No retrieval results found."
            }


        # ======================================================
        # BASIC RERANKER INFORMATION
        # ======================================================

        scores = [

            float(
                result.get(
                    "reranker_score",
                    0.0
                )
            )

            for result in results
        ]

        scores.sort(
            reverse=True
        )


        best_score = scores[0]

        margin = (

            scores[0] - scores[1]

            if len(scores) > 1

            else scores[0]
        )


        # ======================================================
        # KEYWORD COVERAGE
        # ======================================================

        query_words = [

            word.lower()

            for word in re.findall(
                r"\b[a-zA-Z0-9]+\b",
                query
            )

            if len(word) > 2
        ]


        matched_words = set()


        for result in results:

            text = result.get(
                "chunk",
                {}
            ).get(
                "text",
                ""
            ).lower()

            for word in query_words:

                if word in text:

                    matched_words.add(
                        word
                    )


        if query_words:

            keyword_coverage = (

                len(matched_words)
                /
                len(set(query_words))
            )

        else:

            keyword_coverage = 0.0


        # ======================================================
        # INTERFACE-SPECIFIC OVERRIDE
        # ======================================================

        if self._is_interface_query(
            query
        ):

            interface_check = (
                self._check_interface_evidence(
                    query=query,
                    results=results
                )
            )

            if interface_check["supported"]:

                return {

                    "sufficient": True,

                    # Keep original reranker score
                    # for transparency.
                    "best_score": best_score,

                    "margin": margin,

                    "keyword_coverage":
                        keyword_coverage,

                    "reason":
                        interface_check[
                            "reason"
                        ]
                }


        # ======================================================
        # NORMAL EVIDENCE RULES
        # ======================================================

        # ------------------------------------------------------
        # Rule 1: Strong reranker evidence
        # ------------------------------------------------------

        if (
            best_score >= self.min_score
            and
            keyword_coverage >=
            self.min_keyword_coverage
        ):

            return {

                "sufficient": True,

                "best_score": best_score,

                "margin": margin,

                "keyword_coverage":
                    keyword_coverage,

                "reason":
                    "Evidence passed all checks."
            }


        # ------------------------------------------------------
        # Rule 2: Strong semantic result
        # ------------------------------------------------------

        top_result = results[0]

        entity_score = float(
            top_result.get(
                "entity_score",
                0.0
            )
        )

        entity_coverage = float(
            top_result.get(
                "entity_coverage",
                top_result.get(
                    "coverage",
                    0.0
                )
            )
        )

        intent_score = float(
            top_result.get(
                "intent_score",
                0.0
            )
        )

        concept_score = float(
            top_result.get(
                "concept_score",
                0.0
            )
        )


        strong_semantic_evidence = (

            entity_score >= 0.5

            and

            entity_coverage >= 0.5

            and

            (
                intent_score >= 0.7

                or

                concept_score >= 0.7
            )
        )


        if (
            strong_semantic_evidence

            and

            best_score >= 0.5
        ):

            return {

                "sufficient": True,

                "best_score": best_score,

                "margin": margin,

                "keyword_coverage":
                    keyword_coverage,

                "reason":
                    (
                        "Evidence accepted because "
                        "strong semantic/entity/intent "
                        "signals support the result."
                    )
            }


        # ------------------------------------------------------
        # Rule 3: Good score + reasonable margin
        # ------------------------------------------------------

        if (
            best_score >= self.min_score

            and

            margin >= self.min_margin
        ):

            return {

                "sufficient": True,

                "best_score": best_score,

                "margin": margin,

                "keyword_coverage":
                    keyword_coverage,

                "reason":
                    (
                        "Evidence accepted based on "
                        "score and ranking margin."
                    )
            }


        # ======================================================
        # REJECT
        # ======================================================

        if best_score < self.min_score:

            reason = (
                "Best reranker score is too low."
            )

        elif margin < self.min_margin:

            reason = (
                "Score margin between top results "
                "is too small."
            )

        elif (
            keyword_coverage
            <
            self.min_keyword_coverage
        ):

            reason = (
                "Keyword coverage is too low."
            )

        else:

            reason = (
                "Evidence did not satisfy "
                "minimum requirements."
            )


        return {

            "sufficient": False,

            "best_score": best_score,

            "margin": margin,

            "keyword_coverage":
                keyword_coverage,

            "reason": reason
        }