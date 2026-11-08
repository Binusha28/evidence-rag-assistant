"""Evaluate the semantic verifier against labelled examples."""

from evidence_rag.evaluation.nli_cases import NLI_CASES
from evidence_rag.verification.hybrid_verifier import HybridSemanticVerifier


def main() -> None:
    verifier = HybridSemanticVerifier()

    full_correct = 0
    review_candidates = 0

    print("\n=== NLI EVALUATION ===")

    for case in NLI_CASES:
        result = verifier.diagnose(
            claim=case.claim,
            evidence=case.evidence,
        )

        predicted = result.full_passage.label
        correct = predicted == case.expected_label

        if correct:
            full_correct += 1

        if result.needs_context_review:
            review_candidates += 1

        print(f"\nCase: {case.name}")
        print(f"Expected: {case.expected_label}")
        print(f"Predicted: {predicted}")
        print(f"Confidence: {result.full_passage.confidence:.4f}")
        print(f"Correct: {correct}")
        print(f"Context review: {result.needs_context_review}")

        best = result.best_sentence_entailment
        if best is not None:
            print(f"Best sentence entailment: {best.confidence:.4f}")

    supported_cases = [case for case in NLI_CASES if case.expected_label == "entailment"]
    unsupported_cases = [case for case in NLI_CASES if case.expected_label != "entailment"]

    supported_accepted = sum(
        verifier.verify(
            claim=case.claim,
            evidence=case.evidence,
        ).label
        == "entailment"
        for case in supported_cases
    )

    unsupported_rejected = sum(
        verifier.verify(
            claim=case.claim,
            evidence=case.evidence,
        ).label
        != "entailment"
        for case in unsupported_cases
    )

    print("\n=== SAFETY METRICS ===")
    print(
        "Supported-claim acceptance:",
        f"{supported_accepted}/{len(supported_cases)}",
    )
    print(
        "Unsupported-claim rejection:",
        f"{unsupported_rejected}/{len(unsupported_cases)}",
    )

    total = len(NLI_CASES)

    print("\n=== SUMMARY ===")
    print(f"Total cases: {total}")
    print(f"Full-passage correct: {full_correct}/{total}")
    print(f"Full-passage accuracy: {full_correct / total:.2%}")
    print(f"Context-review candidates: {review_candidates}")


if __name__ == "__main__":
    main()
