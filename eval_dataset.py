from datasets import load_dataset


class EvalDataset:
    def __init__(self, tokenizer) -> None:
        self.dataset = load_dataset("Intel/orca_dpo_pairs")["train"]
        self.tokenizer = tokenizer
        self.index = 0

    def convert_to_chatml(self, data):
        if len(data["system"]) > 0:
            message = {"role": "system", "content": data["system"]}
            system = self.tokenizer.apply_chat_template([message], tokenize=False)
        else:
            system = ""

        message = {"role": "user", "content": data["question"]}
        prompt = self.tokenizer.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True
        )

        chosen = data["chosen"] + "<|im_end|>\n"
        rejected = data["rejected"] + "<|im_end|>\n"

        return {
            "prompt": system + prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    def get(self, index):
        data = self.dataset[index]
        return self.convert_to_chatml(data)
