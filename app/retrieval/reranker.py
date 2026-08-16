from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:

    def __init__(self):

        print("Loading cross-encoder reranker...")

        self.model = CrossEncoder(
            MODEL_NAME
        )

        print("Reranker ready.")

    def rerank(
        self,
        query,
        candidates,
        top_k=5
    ):
        """
        Rerank retrieved candidates using
        query-document relevance.
        """

        if not candidates:
            return []

        pairs = []

        for candidate in candidates:

            text = candidate[
                "chunk"
            ]["text"]

            pairs.append(
                (
                    query,
                    text
                )
            )

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores
        ):

            result = candidate.copy()

            result[
                "reranker_score"
            ] = float(score)

            reranked.append(
                result
            )

        reranked.sort(
            key=lambda x: x[
                "reranker_score"
            ],
            reverse=True
        )

        return reranked[:top_k]


def main():

    reranker = Reranker()

    query = (
        "What is the N11 interface?"
    )

    candidates = [
        {
            "chunk": {
                "text":
                "The N11 interface "
                "supports communication "
                "between network functions."
            }
        },
        {
            "chunk": {
                "text":
                "The N2 interface "
                "connects the 5G-AN "
                "and AMF."
            }
        }
    ]

    results = reranker.rerank(
        query,
        candidates,
        top_k=2
    )

    print("\nReranking test:")

    for result in results:

        print(
            result[
                "reranker_score"
            ]
        )

        print(
            result[
                "chunk"
            ]["text"]
        )


if __name__ == "__main__":
    main()