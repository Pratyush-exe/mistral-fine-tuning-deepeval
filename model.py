import torch
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class Mistral:
    def __init__(self, model_id) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config
        )

        self.accelerator = self.load_accelerators()
        self.model = self.load_lora(model)

    def get_model(self):
        return self.model

    def load_lora(self, model):
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, self.lora_config)
        model = self.accelerator.prepare_model(model)

        return model

    def load_accelerators(self):
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(
                offload_to_cpu=True, rank0_only=False
            ),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=False
            ),
        )

        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        return accelerator
