import os
import json

from groq import Groq


class LLMClient:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set."
            )

        self.client = Groq(
            api_key=api_key
        )

        # You can change this later.
        self.model = os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile"
        )

    def generate(self, prompt):

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
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            response_format={
                "type": "json_object"
            }
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        try:

            return json.loads(
                content
            )

        except json.JSONDecodeError as error:

            raise RuntimeError(
                "Groq returned invalid JSON: "
                f"{content}"
            ) from error