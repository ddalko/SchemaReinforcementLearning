import os
import argparse
import json, random

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

from utils import preprocess_dataset
from utils import dLLMTrainer
from utils import dLLMSFTDataset
from utils import dLLMDataCollator

def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    init_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--debugging", action="store_true")
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
    )

    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)

    with open(os.path.join(os.path.dirname(__file__), args.data_file), 'r') as f:
        data = json.load(f)
    
    # Load dataset
    data = data[-10:]
    with open("train_data_sample.json", "w") as f:
        json.dump(data[-1], f, indent=4)
    max_length = model.config.max_sequence_length
    train_data, eval_data = preprocess_dataset(data, tokenizer, max_length)
    print("Train data length: ", len(train_data))
    print("Eval data length: ", len(eval_data))
    train_dataset = dLLMSFTDataset(train_data, tokenizer, max_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, max_length, eval=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=2,
        save_steps=args.save_steps,
        save_total_limit=20,
        load_best_model_at_end=True,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        report_to="wandb" if not args.debugging else "none",
        remove_unused_columns=False,
    )

    num_train_steps = int(
        len(train_dataset)
        * args.num_train_epochs
        / (args.per_device_train_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count())
    )

    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(tokenizer=tokenizer, mask_token_id=126336, max_length=max_length),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train(resume_from_checkpoint=args.resume)