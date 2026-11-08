"""Small labelled dataset for evaluating semantic claim verification."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NLICase:
    name: str
    evidence: str
    claim: str
    expected_label: str


NLI_CASES = (
    NLICase(
        name="exact_feature_map",
        evidence=(
            "A fast R-CNN algorithm performs automatic lesion detection "
            "and disease recognition from fundus images. "
            "The input fundus image is fed into the CNN network to obtain "
            "a feature map. "
            "The derived feature map is used to estimate region proposals "
            "(candidate lesion regions in squared boxes), which are then "
            "classified and predicted as different disease categories."
        ),
        claim=("The input fundus image is fed into the CNN network to obtain a feature map."),
        expected_label="entailment",
    ),
    NLICase(
        name="paraphrased_feature_map",
        evidence=("The input fundus image is fed into the CNN network to obtain a feature map."),
        claim=("A CNN extracts features from retinal fundus images to generate a feature map."),
        expected_label="entailment",
    ),
    NLICase(
        name="lesion_region_detection",
        evidence=(
            "The derived feature map is used to estimate region proposals "
            "(candidate lesion regions in squared boxes)."
        ),
        claim=("The feature map is used to identify candidate lesion regions."),
        expected_label="entailment",
    ),
    NLICase(
        name="unsupported_accuracy",
        evidence=("The CNN generates feature maps from retinal fundus images."),
        claim=("The CNN achieves 100% diagnostic accuracy."),
        expected_label="neutral",
    ),
    NLICase(
        name="explicit_contradiction",
        evidence="The model achieved 85% accuracy.",
        claim="The model achieved 100% accuracy.",
        expected_label="contradiction",
    ),
    NLICase(
        name="unsupported_clinical_claim",
        evidence=("The CNN identifies candidate lesion regions in retinal fundus images."),
        claim=("The CNN eliminates the need for ophthalmologists."),
        expected_label="neutral",
    ),
    NLICase(
        name="exact_lesion_claim",
        evidence=("The derived feature map is used to estimate candidate lesion regions."),
        claim=("The derived feature map is used to estimate candidate lesion regions."),
        expected_label="entailment",
    ),
    NLICase(
        name="unsupported_combined_claim",
        evidence=("The CNN generates feature maps from fundus images."),
        claim=("The CNN generates feature maps from fundus images and achieves 100% accuracy."),
        expected_label="neutral",
    ),
)
