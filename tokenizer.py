from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model_id) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, model_max_length=512, padding_side="left", add_eos_token=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            add_bos_token=True,
        )

    def tokenize(self, prompt):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result
