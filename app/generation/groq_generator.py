import os

from groq import Groq

from app.generation.prompt import (
    SYSTEM_PROMPT,
    build_prompt
)


class GroqGenerator:

    def __init__(
        self,
        model="llama-3.3-70b-versatile"
    ):

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "GROQ_API_KEY environment variable "
                "is not set."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = model

        print(
            f"Groq generator ready: {model}"
        )


    def generate(
        self,
        query,
        results
    ):

        prompt = build_prompt(
            query,
            results
        )

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.1,

            max_tokens=1500
        )

        answer = response.choices[0].message.content

        return answer