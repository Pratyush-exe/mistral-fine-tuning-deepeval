from eval_dataset import EvalDataset
from tokenizer import Tokenizer
from model import Mistral
from trl import DPOTrainer
from transformers import TrainingArguments

model_id = "mistralai/Mistral-7B-v0.1"

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    save_strategy="no",
    logging_steps=1,
    output_dir="./",
    optim="paged_adamw_32bit",
    warmup_steps=100,
    bf16=True,
)

model = Mistral(model_id)
tokenizer = Tokenizer(model_id)
dataset = EvalDataset(tokenizer)


dpo_trainer = DPOTrainer(
    model,
    model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=model.lora_config,
    beta=0.1,
    max_prompt_length=1024,
    max_length=1536,
)

dpo_trainer.train()
