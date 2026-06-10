try:
    from curator_base import BaseDatasetCurator
except ModuleNotFoundError:
    from tools.data_curation.curator_base import BaseDatasetCurator

class UltraChatCurator(BaseDatasetCurator):
    def __init__(self, split="train_sft"):
        super().__init__(
            dataset_path="HuggingFaceH4/ultrachat_200k",
            config_name=None,
            split=split,
            text_key="messages",
            score_key=None,
            alias="ultrachat"
        )
