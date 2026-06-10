try:
    from curator_base import BaseDatasetCurator
except ModuleNotFoundError:
    from tools.data_curation.curator_base import BaseDatasetCurator

class FineMathCurator(BaseDatasetCurator):
    def __init__(self, split="train"):
        super().__init__(
            dataset_path="HuggingFaceTB/finemath",
            config_name="finemath-4plus",
            split=split,
            text_key="text",
            score_key="score",
            alias="finemath"
        )
