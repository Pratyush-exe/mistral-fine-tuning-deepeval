from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model_id) -> None:
        chat_template = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta"
        ).chat_template

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            model_max_length=512,
            padding_side="left",
            add_eos_token=True,
            chat_template=chat_template,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            model_id, add_bos_token=True, chat_template=chat_template
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
