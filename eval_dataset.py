from datasets import load_dataset


class EvalDataset:
    def __init__(self, tokenizer) -> None:
        self.dataset = load_dataset("Intel/orca_dpo_pairs")["train"]
        self.tokenizer = tokenizer
        self.index = 0

    def convert_to_chatml(self, data):
        messages = []
        if len(data["system"]) > 0:
            message = {"role": "system", "content": data["system"]}
            messages.append(message)

        message = {"role": "user", "content": data["question"]}
        messages.append(message)

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        chosen = data["chosen"] + "<|im_end|>\n"
        rejected = data["rejected"] + "<|im_end|>\n"

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    def get(self, index, apply_template=True):
        data = self.dataset[index]
        if apply_template:
            return self.convert_to_chatml(data)
        else:
            return data

    def create_dataset(self):
        self.dataset = self.dataset.map(
            self.convert_to_chatml, remove_columns=["system", "question"]
        )
