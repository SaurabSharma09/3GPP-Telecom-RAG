from app.llm.client import LLMClient


class SemanticEvidenceVerifier:

    def __init__(self):

        self.llm = LLMClient()


    def verify(
        self,
        query,
        results
    ):

        if not results:

            return {
                "verdict": "NOT_SUPPORTED",
                "confidence": 1.0,
                "reason": "No evidence was retrieved."
            }


        evidence_blocks = []


        for index, result in enumerate(
            results,
            start=1
        ):

            chunk = result["chunk"]

            metadata = chunk["metadata"]

            specification = metadata.get(
                "specification",
                "Unknown"
            )

            section_number = metadata.get(
                "section_number"
            )

            parent_section = metadata.get(
                "parent_section"
            )

            section_title = metadata.get(
                "section_title",
                "Unknown"
            )


            if section_number:

                section = (
                    f"{section_number} "
                    f"{section_title}"
                )

            elif parent_section:

                section = (
                    f"{parent_section} → "
                    f"{section_title}"
                )

            else:

                section = section_title


            source = (
                f"{specification} | "
                f"Section: {section}"
            )


            evidence_blocks.append(
                f"""
========================
EVIDENCE {index}
========================

SOURCE:
{source}

TEXT:
{chunk["text"]}
"""
            )


        evidence_text = "\n".join(
            evidence_blocks
        )


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
   telecom, 3GPP, AMF, SMF, N11, or any
   other subject.

3. Merely mentioning an entity is NOT
   sufficient evidence.

4. If the question asks for a definition,
   the evidence should actually define or
   describe the entity.

5. If the question asks for a role,
   the evidence should describe functions,
   responsibilities, or behavior.

6. If the question asks for a procedure,
   the evidence should describe that procedure.

7. If the evidence supports only part of
   the question, use PARTIALLY_SUPPORTED.

8. If the evidence does not actually support
   the question, use NOT_SUPPORTED.

9. Be conservative. When uncertain, prefer
   PARTIALLY_SUPPORTED or NOT_SUPPORTED.

10. Do not infer facts that are not explicitly
    supported by the evidence.


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


        result = self.llm.generate(
            prompt
        )


        verdict = result.get(
            "verdict",
            "NOT_SUPPORTED"
        )

        confidence = result.get(
            "confidence",
            0.0
        )

        reason = result.get(
            "reason",
            ""
        )


        if verdict not in {
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
            "NOT_SUPPORTED"
        }:

            verdict = "NOT_SUPPORTED"


        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError
        ):

            confidence = 0.0


        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )


        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason
        }