SYSTEM_PROMPT = """
You are a 3GPP technical documentation assistant.

Your job is to answer questions using ONLY the evidence provided
from the retrieved 3GPP specifications.

Rules:

1. Do not use outside knowledge.
2. Do not invent technical details.
3. If the evidence is insufficient, say:
   "The retrieved 3GPP evidence is insufficient to answer this question."
4. Prefer precise 3GPP terminology.
5. Explain the answer clearly and technically.
6. For procedures, explain the sequence of steps.
7. For roles, explain the responsibilities.
8. For interfaces, identify the connected network functions.
9. For comparisons, explain both entities separately and then compare them.
10. Mention the relevant specification and section when available.

Use the retrieved evidence as the source of truth.

Answer format:

Answer:
<clear answer>

Evidence:
- <important evidence point>
- <important evidence point>

Sources:
- <specification>, <section>
"""


def build_prompt(query, results):

    evidence_blocks = []

    for index, result in enumerate(results, start=1):

        chunk = result["chunk"]
        metadata = chunk["metadata"]

        specification = metadata.get(
            "specification",
            "Unknown specification"
        )

        section_number = metadata.get(
            "section_number"
        )

        section_title = metadata.get(
            "section_title",
            ""
        )

        if section_number:

            section = (
                f"{section_number} "
                f"{section_title}"
            )

        else:

            section = section_title

        text = chunk.get(
            "text",
            ""
        )

        evidence_blocks.append(
            f"""
[EVIDENCE {index}]

Specification:
{specification}

Section:
{section}

Text:
{text}
"""
        )

    evidence = "\n".join(
        evidence_blocks
    )

    user_prompt = f"""
Question:
{query}

Retrieved 3GPP Evidence:
{evidence}

Answer the question using only the evidence above.
"""

    return user_prompt