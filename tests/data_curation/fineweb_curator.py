try:
    from curator_base import BaseDatasetCurator
except ModuleNotFoundError:
    from tests.data_curation.curator_base import BaseDatasetCurator

class FineWebCurator(BaseDatasetCurator):
    def __init__(self, split="train"):
        super().__init__(
            dataset_path="HuggingFaceFW/fineweb-edu",
            config_name="sample-10BT",
            split=split,
            text_key="text",
            score_key="score",
            alias="fineweb-edu"
        )

    def evaluate_score_threshold(self, threshold=3.0, sample_size=20):
        """Analyzes document retention rate under specific curriculum score thresholds."""
        samples = self.fetch_samples(sample_size)
        if not samples:
            return 0.0

        retained = 0
        total_score = 0.0
        scores_seen = []

        for item in samples:
            score = float(item.get(self.score_key, 0.0))
            scores_seen.append(score)
            if score >= threshold:
                retained += 1

        retention_rate = (retained / sample_size) * 100
        print(f"📊 FineWeb Threshold Evaluation (Threshold >= {threshold}):")
        print(f"   - Total Samples Analyzed: {sample_size}")
        print(f"   - Retained Documents:     {retained} ({retention_rate:.1f}%)")
        if scores_seen:
            print(f"   - Score range:            Min {min(scores_seen):.2f} / Max {max(scores_seen):.2f}")
        return retention_rate
