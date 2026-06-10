try:
    from curator_base import BaseDatasetCurator
except ModuleNotFoundError:
    from tools.data_curation.curator_base import BaseDatasetCurator

class CodeCurator(BaseDatasetCurator):
    def __init__(self, split="train"):
        super().__init__(
            dataset_path="TokenBender/code_instructions_122k_alpaca_style",
            config_name=None,
            split=split,
            text_key="text",
            score_key=None,
            alias="code_instructions"
        )

    def extract_text(self, item):
        """Custom extraction to fallback to manual alpaca-style formatting if 'text' is missing."""
        text = super().extract_text(item)
        if text:
            return text
            
        instruction = item.get("instruction", "")
        input_data = item.get("input", "")
        output = item.get("output", "")
        
        formatted = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}"
        if input_data:
            formatted += f"\n\n### Input:\n{input_data}"
        formatted += f"\n\n### Response:\n{output}"
        return formatted
