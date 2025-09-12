import torch

from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

lora_path     = "./lora_checkpoint"
save_path     = "./merged-llama3-8b"

model_name = "/models/LLaDA-8B-Instruct"
lora_path = "/workspace/SchemaReinforcementLearning/train/results/LLaDA-8B-sft/checkpoint-10320"
save_path = "/workspace/SchemaReinforcementLearning/train/results/LLaDA-8B-sft/merged"

tokenizer = AutoTokenizer.from_pretrained(
    model_name, padding_side="right", trust_remote_code=True, use_fast=True
)

base = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base, lora_path)
merged = model.merge_and_unload()
merged.save_pretrained(save_path, safe_serialization=True) 
tokenizer.save_pretrained(save_path)
