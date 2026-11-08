"""Inspect NLI label mapping and test simple entailment examples."""

import numpy as np

from evidence_rag.verification.semantic_verifier import SemanticClaimVerifier


def main() -> None:
    verifier = SemanticClaimVerifier(
        model_name="cross-encoder/nli-deberta-v3-base",
        device="cpu",
    )

    print("\n=== MODEL LABEL MAPPING ===")
    print(verifier.model.config.id2label)

    examples = [
        (
            "A retinal fundus image is passed through a CNN to generate a feature map.",
            "A CNN generates a feature map from a retinal image.",
        ),
        (
            "Vision Transformers dominate ResNets on the performance/compute trade-off.",
            "Vision Transformers outperform ResNets on performance-compute trade-offs.",
        ),
        (
            "Vision Transformers dominate ResNets on the performance/compute trade-off.",
            "Vision Transformers always require more compute "
            "than ResNets for the same performance.",
        ),
    ]

    for index, (premise, hypothesis) in enumerate(
        examples,
        start=1,
    ):
        scores = np.asarray(
            verifier.model.predict(
                [(premise, hypothesis)],
                show_progress_bar=False,
                apply_softmax=True,
            )
        ).reshape(-1)

        result = verifier.verify(
            claim=hypothesis,
            evidence=premise,
        )

        print(f"\n=== EXAMPLE {index} ===")
        print("Premise:", premise)
        print("Hypothesis:", hypothesis)
        print("Scores:", scores)
        print("Prediction:", result.label)
        print("Confidence:", round(result.confidence, 4))


if __name__ == "__main__":
    main()
