import os
import json

from groq import Groq


class LLMClient:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:

            raise RuntimeError("GROQ_API_KEY environment variable " "is not set.")

        self.client = Groq(api_key=api_key)

        # Use GROQ_MODEL when explicitly configured.
        # Otherwise use the current supported model.
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

        print(f"LLMClient ready: {self.model}")

    def generate(self, prompt):

        if not prompt or not str(prompt).strip():

            raise ValueError("Prompt cannot be empty.")

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evidence verification "
                        "system. Use ONLY the evidence supplied "
                        "in the user message. Never use outside "
                        "knowledge. Never invent facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        if not response.choices:

            raise RuntimeError("Groq returned no completion choices.")

        content = response.choices[0].message.content

        if not content or not content.strip():

            raise RuntimeError("Groq returned an empty response.")

        try:

            return json.loads(content)

        except json.JSONDecodeError as error:

            raise RuntimeError("Groq returned invalid JSON: " f"{content}") from error
