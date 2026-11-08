"""Compare NLI models using the same labelled evaluation cases."""

from evidence_rag.evaluation.nli_cases import NLI_CASES
from evidence_rag.verification.semantic_verifier import (
    SemanticClaimVerifier,
)

MODELS = (
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-deberta-v3-base",
)


def evaluate_model(model_name: str) -> dict[str, object]:
    """Evaluate one model without changing production settings."""
    print(f"\n=== MODEL: {model_name} ===", flush=True)

    verifier = SemanticClaimVerifier(model_name=model_name)

    correct = 0
    supported_total = 0
    supported_accepted = 0
    unsupported_total = 0
    unsupported_rejected = 0

    for case in NLI_CASES:
        result = verifier.verify(
            claim=case.claim,
            evidence=case.evidence,
        )

        if result.label == case.expected_label:
            correct += 1

        if case.expected_label == "entailment":
            supported_total += 1
            if result.label == "entailment":
                supported_accepted += 1
        else:
            unsupported_total += 1
            if result.label != "entailment":
                unsupported_rejected += 1

        print(
            f"{case.name:30} "
            f"expected={case.expected_label:14} "
            f"predicted={result.label:14} "
            f"confidence={result.confidence:.4f}",
            flush=True,
        )

    total = len(NLI_CASES)

    summary = {
        "model": model_name,
        "accuracy": correct / total,
        "supported_acceptance": supported_accepted / supported_total,
        "unsupported_rejection": unsupported_rejected / unsupported_total,
    }

    print("\nSUMMARY")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Supported acceptance: {summary['supported_acceptance']:.2%}")
    print(f"Unsupported rejection: {summary['unsupported_rejection']:.2%}")

    return summary


def main() -> None:
    results = []

    for model_name in MODELS:
        results.append(evaluate_model(model_name))

    print("\n=== FINAL COMPARISON ===")

    for result in results:
        print(
            f"{result['model']}\n"
            f"  Accuracy: {result['accuracy']:.2%}\n"
            f"  Supported acceptance: "
            f"{result['supported_acceptance']:.2%}\n"
            f"  Unsupported rejection: "
            f"{result['unsupported_rejection']:.2%}"
        )


if __name__ == "__main__":
    main()
