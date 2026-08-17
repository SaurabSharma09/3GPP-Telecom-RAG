import re

from app.llm.client import LLMClient


# Keep the semantic-verification request small enough for
# the current Groq token-per-minute limit.
VERIFIER_TOP_K = 5


class SemanticEvidenceVerifier:
    """
    Semantic evidence verifier.

    Primary decision:
        LLM-based evidence verification

    Deterministic safeguard:
        When the retrieved evidence contains an explicit, strong
        canonical/technical match, do not allow the LLM verifier
        to incorrectly downgrade clearly supported evidence.
    """

    def __init__(self):

        self.llm = LLMClient()

    # ==========================================================
    # INTERFACE DETECTION
    # ==========================================================

    @staticmethod
    def extract_interface(query):

        match = re.search(r"\b(N\d+[A-Za-z]?)\b", query, re.IGNORECASE)

        if match:
            return match.group(1).upper()

        return None

    # ==========================================================
    # DETERMINISTIC SUPPORT CHECK
    # ==========================================================

    def deterministic_support(self, query, results):
        """
        Detect strong explicit evidence that can safely protect against
        an LLM false-negative.

        This does NOT answer the question from outside knowledge.
        It only checks signals already present in retrieved metadata/text.
        """

        if not results:
            return None

        query_lower = query.lower().strip()

        # ------------------------------------------------------
        # Strong canonical-section evidence
        # ------------------------------------------------------

        for result in results:

            chunk = result.get("chunk", {})

            metadata = chunk.get("metadata", {})

            text = chunk.get("text", "")

            canonical_score = float(result.get("canonical_concept_score", 0.0))

            entity_score = float(result.get("entity_score", 0.0))

            coverage = float(result.get("entity_coverage", result.get("coverage", 0.0)))

            section_number = str(metadata.get("section_number", ""))

            section_title = str(metadata.get("section_title", ""))

            section_lower = (f"{section_number} " f"{section_title}").lower()

            text_lower = text.lower()

            # --------------------------------------------------
            # Registration
            # --------------------------------------------------

            is_registration_query = (
                "registration" in query_lower or "register" in query_lower
            )

            registration_evidence = ("4.2.2.2.2" in section_number) and (
                "general registration" in section_lower
            )

            if (
                is_registration_query
                and registration_evidence
                and canonical_score >= 0.8
            ):

                return {
                    "verdict": "SUPPORTED",
                    "confidence": 0.95,
                    "reason": (
                        "Canonical 3GPP General Registration "
                        "evidence was retrieved and explicitly "
                        "matches the registration query."
                    ),
                }

            # --------------------------------------------------
            # PDU Session Establishment
            # --------------------------------------------------

            is_pdu_query = "pdu session establishment" in query_lower

            pdu_canonical = section_number in {"4.3.2.1", "4.3.2.2.1"}

            if is_pdu_query and pdu_canonical and canonical_score >= 0.8:

                return {
                    "verdict": "SUPPORTED",
                    "confidence": 0.95,
                    "reason": (
                        "Canonical 3GPP PDU Session "
                        "Establishment evidence was retrieved."
                    ),
                }

            # --------------------------------------------------
            # Network-function role
            # --------------------------------------------------

            role_match = re.search(
                r"\b(role|responsibilit(?:y|ies)|function)\b", query_lower
            )

            if role_match:

                canonical_role_sections = {
                    "6.2.1": "amf",
                    "6.2.2": "smf",
                    "6.2.3": "upf",
                }

                for canonical_section, entity in canonical_role_sections.items():

                    entity_present = re.search(rf"\b{re.escape(entity)}\b", query_lower)

                    if (
                        entity_present
                        and section_number == canonical_section
                        and canonical_score >= 0.8
                        and coverage >= 0.5
                        and entity_score >= 0.5
                    ):

                        return {
                            "verdict": "SUPPORTED",
                            "confidence": 0.95,
                            "reason": (
                                f"Canonical 3GPP {entity.upper()} "
                                "functional-description evidence "
                                "was retrieved."
                            ),
                        }

            # --------------------------------------------------
            # Direct interface evidence
            # --------------------------------------------------

            interface = self.extract_interface(query)

            if interface:

                exact_interface = bool(
                    re.search(rf"\b{re.escape(interface)}\b", text_lower, re.IGNORECASE)
                )

                # Reference-point language is especially strong.
                reference_point_evidence = "reference point" in text_lower

                # Common explicit relationship wording.
                relationship_evidence = (
                    "smf" in text_lower and "upf" in text_lower
                ) or ("amf" in text_lower and "smf" in text_lower)

                interface_section = (
                    "interface" in section_lower
                    or "reference point" in section_lower
                    or "protocol stack" in section_lower
                )

                if (
                    exact_interface
                    and (
                        reference_point_evidence
                        or relationship_evidence
                        or interface_section
                    )
                    and (coverage >= 0.5 or entity_score >= 0.5)
                ):

                    return {
                        "verdict": "SUPPORTED",
                        "confidence": 0.95,
                        "reason": (
                            f"Retrieved 3GPP evidence explicitly "
                            f"describes {interface} and its network "
                            "function relationship."
                        ),
                    }

        return None

    # ==========================================================
    # VERIFY
    # ==========================================================

    def verify(self, query, results):

        if not results:

            return {
                "verdict": "NOT_SUPPORTED",
                "confidence": 1.0,
                "reason": "No evidence was retrieved.",
            }

        # ------------------------------------------------------
        # Only send strongest evidence to Groq.
        # ------------------------------------------------------

        results = results[:VERIFIER_TOP_K]

        # ------------------------------------------------------
        # Build evidence blocks
        # ------------------------------------------------------

        evidence_blocks = []

        for index, result in enumerate(results, start=1):

            chunk = result.get("chunk", {})

            metadata = chunk.get("metadata", {})

            specification = metadata.get("specification", "Unknown")

            section_number = metadata.get("section_number")

            parent_section = metadata.get("parent_section")

            section_title = metadata.get("section_title", "Unknown")

            if section_number:

                section = f"{section_number} " f"{section_title}"

            elif parent_section:

                section = f"{parent_section} → " f"{section_title}"

            else:

                section = section_title or "Unknown section"

            source = f"{specification} | " f"Section: {section}"

            text = chunk.get("text", "").strip()

            evidence_blocks.append(
                f"""
========================
EVIDENCE {index}
========================

SOURCE:
{source}

TEXT:
{text}
"""
            )

        evidence_text = "\n".join(evidence_blocks)

        # ------------------------------------------------------
        # Semantic verification prompt
        # ------------------------------------------------------

        prompt = f"""
You are a strict evidence-grounding verifier
for a telecom 3GPP RAG system.

Your job is NOT to answer the question.

Your job is ONLY to determine whether the
retrieved evidence actually supports the question.

QUESTION:
{query}

RETRIEVED EVIDENCE:
{evidence_text}

RULES:

1. Use ONLY the supplied evidence.

2. Do NOT use your own knowledge about
   telecom, 3GPP, AMF, SMF, N11, N4,
   registration, or any other subject.

3. Merely mentioning an entity is NOT
   sufficient evidence.

4. If the question asks for a definition,
   the evidence should actually define or
   describe the entity.

5. If the question asks for a role,
   the evidence should describe functions,
   responsibilities, or behavior.

6. If the question asks for an interface,
   explicit evidence identifying the
   interface/reference point and the
   connected network functions is sufficient.

7. If the question asks for a procedure,
   the evidence should describe that procedure.

8. If the retrieved evidence contains a
   canonical section specifically addressing
   the question, treat that as strong evidence.

9. If the evidence supports only part of
   the question, use PARTIALLY_SUPPORTED.

10. If the evidence does not actually support
    the question, use NOT_SUPPORTED.

11. Be conservative. Do not invent facts.

12. Do not infer technical details that are
    absent from the supplied evidence.

Return ONLY JSON:

{{
    "verdict": "SUPPORTED",
    "confidence": 0.0,
    "reason": "Brief explanation based only on the evidence."
}}

Allowed verdicts:

SUPPORTED
PARTIALLY_SUPPORTED
NOT_SUPPORTED

Confidence must be between 0.0 and 1.0.
"""

        # ------------------------------------------------------
        # LLM verification
        # ------------------------------------------------------

        result = self.llm.generate(prompt)

        verdict = result.get("verdict", "NOT_SUPPORTED")

        confidence = result.get("confidence", 0.0)

        reason = result.get("reason", "")

        if verdict not in {"SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"}:

            verdict = "NOT_SUPPORTED"

        try:

            confidence = float(confidence)

        except (TypeError, ValueError):

            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        # ------------------------------------------------------
        # Deterministic protection against false negatives
        # ------------------------------------------------------
        #
        # Only override a negative/partial LLM verdict when the
        # retrieved metadata/text independently provides a strong,
        # explicit technical match.
        # ------------------------------------------------------

        deterministic = self.deterministic_support(query=query, results=results)

        if deterministic is not None:

            if verdict in {"NOT_SUPPORTED", "PARTIALLY_SUPPORTED"}:

                return deterministic

        # ------------------------------------------------------
        # Normal LLM verdict
        # ------------------------------------------------------

        return {"verdict": verdict, "confidence": confidence, "reason": reason}
