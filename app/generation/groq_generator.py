import os

from groq import Groq

from app.generation.prompt import SYSTEM_PROMPT, build_prompt


# Keep the final generation request small enough for the
# current Groq token-per-minute limit.
GENERATOR_TOP_K = 5


class GroqGenerator:
    """
    Generates the final answer only from verified retrieved evidence.

    The pipeline is responsible for deciding whether generation is allowed.
    This class is responsible only for constructing the prompt and calling
    the Groq model.
    """

    def __init__(self, model="openai/gpt-oss-120b"):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:

            raise ValueError("GROQ_API_KEY environment variable " "is not set.")

        self.client = Groq(api_key=api_key)

        self.model = model

        print(f"Groq generator ready: {model}")

    def generate(self, query, results):
        """
        Generate a grounded answer from the strongest verified
        retrieved evidence.

        Expected pipeline conditions:
            Evidence Gate      -> PASS
            Lexical Verifier   -> SUPPORTED
            Semantic Verifier  -> SUPPORTED
        """

        if not query or not str(query).strip():

            raise ValueError("Query cannot be empty.")

        if not results:

            return (
                "The retrieved 3GPP evidence is "
                "insufficient to answer this question."
            )

        # --------------------------------------------------------------
        # Limit evidence sent to Groq.
        #
        # Hybrid retrieval -> 20
        # Reranking        -> 15
        # Generation       -> 5
        #
        # This keeps the request within the current TPM limit.
        # --------------------------------------------------------------

        results = results[:GENERATOR_TOP_K]

        prompt = build_prompt(query, results)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            # Concise technical answer is enough for the assignment.
            max_tokens=1000,
        )

        if not response.choices:

            raise RuntimeError("Groq returned no completion choices.")

        answer = response.choices[0].message.content

        if not answer or not answer.strip():

            raise RuntimeError("Groq returned an empty answer.")

        return answer.strip()
