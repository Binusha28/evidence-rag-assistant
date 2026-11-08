import numpy as np

from evidence_rag.retrieval.embeddings import EmbeddingModel


def main() -> None:
    model = EmbeddingModel()

    sentences = [
        "Deep learning helps analyse medical images.",
        "Neural networks can examine healthcare scans.",
        "The weather is sunny today.",
    ]

    embeddings = model.encode(sentences)

    similarity_matrix = embeddings @ embeddings.T

    print("\nSEMANTIC SIMILARITY RESULTS")
    print("-" * 40)

    for index, sentence in enumerate(sentences):
        print(f"{index + 1}. {sentence}")

    print("\nCosine similarity matrix:")
    print(np.round(similarity_matrix, 3))

    healthcare_similarity = float(similarity_matrix[0, 1])
    unrelated_similarity = float(similarity_matrix[0, 2])

    print(f"\nHealthcare similarity: {healthcare_similarity:.3f}")
    print(f"Unrelated similarity: {unrelated_similarity:.3f}")

    if healthcare_similarity > unrelated_similarity:
        print("\nSUCCESS: Related sentences have higher similarity.")
    else:
        print("\nRESULT: The expected similarity ordering was not observed.")


if __name__ == "__main__":
    main()
