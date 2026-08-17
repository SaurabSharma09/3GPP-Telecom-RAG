SYSTEM_PROMPT = """
You are a 3GPP technical documentation assistant.

Your job is to answer the user's question using ONLY the
retrieved 3GPP evidence provided to you.

The retrieved evidence is the only source of truth.

STRICT GROUNDING RULES:

1. Do not use outside knowledge, memory, assumptions, or general
   knowledge about telecom systems.

2. Do not invent, infer, or fill in missing technical details.

3. Every factual claim in the answer must be supported by the
   retrieved evidence.

4. If the retrieved evidence does not sufficiently support the
   question, do NOT answer from your own knowledge.

5. In that case, respond exactly with:

   "The retrieved 3GPP evidence is insufficient to answer this question."

6. Do not combine unrelated evidence blocks to create a conclusion
   that is not explicitly supported by the retrieved material.

7. Prefer precise terminology used by the 3GPP specifications.

8. Keep the answer concise and technically clear.

9. For procedures, describe the sequence only when the retrieved
   evidence contains the corresponding steps.

10. For network-function role questions, describe responsibilities
    only when supported by the evidence.

11. For interface questions, identify connected network functions
    only when the evidence explicitly supports the relationship.

12. For comparison questions, explain each entity separately and
    compare them using only retrieved evidence.

13. Cite the relevant specification and section for every source used.

14. Never cite a section merely because its title looks relevant.
    The cited evidence must actually support the statement.

ANSWER FORMAT:

Answer:
<clear, evidence-grounded answer>

Evidence:
- <important supported evidence point>
- <important supported evidence point>

Sources:
- <specification>, <section number>, <section title>
"""


def build_prompt(query, results):

    evidence_blocks = []

    for index, result in enumerate(results, start=1):

        chunk = result.get("chunk", {})

        metadata = chunk.get("metadata", {})

        specification = metadata.get("specification", "Unknown specification")

        section_number = metadata.get("section_number")

        section_title = metadata.get("section_title", "")

        if section_number and section_title:

            section = f"{section_number} " f"{section_title}"

        elif section_number:

            section = str(section_number)

        else:

            section = section_title or "Unknown section"

        section_path = metadata.get("section_path", [])

        text = chunk.get("text", "").strip()

        evidence_blocks.append(
            f"""
[EVIDENCE {index}]

Specification:
{specification}

Section:
{section}

Section Path:
{section_path}

Evidence Text:
{text}
"""
        )

    evidence = "\n".join(evidence_blocks)

    return f"""
Question:
{query}

Retrieved 3GPP Evidence:
{evidence}

IMPORTANT:
Answer ONLY from the retrieved evidence above.
Do not use outside knowledge.
Do not infer unsupported facts.
If the evidence is insufficient, use the exact
insufficient-evidence response specified by the system instructions.
"""
