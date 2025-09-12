from trl import SFTConfig, SFTTrainer
from datasets import Dataset 
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, random
from transformers import (
    DataCollatorForLanguageModeling,
)
import numpy as np
import warnings
from typing import Any, Dict, List, Optional, Union

import argparse
import os

random.seed(42)

def formatting_prompts_func(example):
    # use llama3 prompt template
    batch_messages = example["messages"]
    formatted_prompts = []
    for messages in batch_messages:
        formatted_prompt = '<|begin_of_text|>'
        for message in messages:
            if message['role'] == 'system':
                formatted_prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{message['content']}<|eot_id|>"
            elif message['role'] == 'user':
                formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message['content']}<|eot_id|>"
            elif message['role'] == 'assistant':
                formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{message['content']}<|eot_id|>"
            elif message['role'] == 'tool':
                formatted_prompt += f"<|start_header_id|>ipython<|end_header_id|>\n\n{message['content']}<|eot_id|>"
            else:
                raise ValueError(f"invalid role: {message['role']}")
        formatted_prompts.append(formatted_prompt)
    return formatted_prompts

class DataCollatorForToolCall(DataCollatorForLanguageModeling):
    """
    Data collator used for multi turn chat tasks that containing tool calls. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant (also tool calls).

    Args:
        response_template (`List[str]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. As a list, both simple text response and tool calls can be accepted.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets.  It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: List[str],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_templates = response_template
        if isinstance(response_template, list):
            self.response_token_ids = []
            for r in response_template:
                self.response_token_ids.append(self.tokenizer.encode(r, add_special_tokens=False))
        else:
            raise Exception('Wrong response_template provided in DataCollatorForToolCall')

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        response_token_ids_start_none_cnt = 0

        if self.instruction_template is None:
            for i in range(len(examples)):
                for response_token_ids in self.response_token_ids:
                    response_token_ids_start_idx = None

                    for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                        # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                        if (
                            response_token_ids
                            == batch["labels"][i][idx : idx + len(response_token_ids)].tolist()
                        ):
                            response_token_ids_start_idx = idx

                    if response_token_ids_start_idx is None:
                        # warnings.warn(
                        #     f"Could not find response key `{self.response_templates}` in the "
                        #     f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        #     f"This instance will be ignored in loss calculation. "
                        #     f"Note, if this happens often, consider increasing the `max_seq_length`."
                        # )
                        response_token_ids_start_none_cnt += 1
                    else:
                        response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)

                        # Make pytorch loss function ignore all tokens up through the end of the response key
                        batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index
                if response_token_ids_start_none_cnt == len(self.response_token_ids):
                    warnings.warn(
                        f"Could not find any response key in `{self.response_templates}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

        else:
            # TODO
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    finetune_right_pad_token = "<|finetune_right_pad_id|>"
    tokenizer.pad_token = finetune_right_pad_token

    with open(os.path.join(os.path.dirname(__file__), args.data_file), 'r') as f:
        data = json.load(f)
    print(len(data))
    random.shuffle(data)
    train_data = data[:int(len(data)*0.9)]
    eval_data = data[int(len(data)*0.9):]
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    instruction_template = "<|start_header_id|>user<|end_header_id|>\n\n"
    collator = DataCollatorForToolCall(
        response_template=[
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|start_header_id|>ipython<|end_header_id|>\n\n"
        ],
        instruction_template=None,
        tokenizer=tokenizer
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    training_args = SFTConfig(
        output_dir=os.path.join(os.path.dirname(__file__), args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        per_device_eval_batch_size=1,
        save_steps=1000,
        max_seq_length=args.max_seq_length,
        bf16=True,
        deepspeed="train/sft/ds_z3_config.json", # from LLaMA-Factory
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # formatting_func=formatting_prompts_func,
        args=training_args,
        data_collator=collator,
    )
    trainer.train()