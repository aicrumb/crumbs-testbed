# this was hastily put together from a notebook, there might be bugs :(
import argparse
import json
import os
from itertools import chain

import torch
import transformers
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

torch.manual_seed(0)

# i use this class in all my code to make it look nicer
# makes sure you can do config.value instead of config["value"] which looks stupid and gross
class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def main(args):
    # kinda weird patching over notebook
    config_json = json.loads(open(args.config, "r").read())
    train_config = Config(**config_json)
    train_config.torch_dtype = torch.bfloat16 if train_config.bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_id)
    tokenizer.model_max_length = train_config.context_length
    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_id,
        device_map='auto',  # device_map={"":0}, # entire model needs to be on cuda for lora, make sure it'll fit
        load_in_8bit=train_config.load_in_8bit,
        load_in_4bit=train_config.load_in_4bit,
        trust_remote_code=True,
        torch_dtype=train_config.torch_dtype,
    )

    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    if train_config.use_lora:
        config = LoraConfig(
            r=train_config.lora["rank"],
            lora_alpha=train_config.lora["alpha"],
            target_modules=train_config.lora["target_modules"],
            lora_dropout=train_config.lora["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        #for name, module in model.named_modules():
            #if "norm" in name:
                #module = module.to(torch.float32)
                
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    data = load_dataset(**train_config.dataset_kwargs)
    if train_config.cluster is not None and train_config.cluster != "none":
        data = data.filter(lambda x: x["cluster"] == train_config.cluster)
    tokenizer.model_max_length = train_config.context_length

    def alpaca(examples, tokenizer, config):
        instructions = examples[config.dataset_instruction_column]
        instruction_roles = [config.dataset_user_role] * len(instructions)
        inputs = (
            examples[config.dataset_input_column]
            if config.dataset_input_column not in [None, "none"]
            else None
        )
        if inputs is not None:
            input_roles = [config.dataset_input_role] * len(inputs)
        outputs = examples[config.dataset_output_column]
        output_roles = [config.dataset_assistant_role] * len(outputs)

        if inputs is not None:
            prompts = zip(
                instruction_roles,
                instructions,
                input_roles,
                inputs,
                output_roles,
                outputs,
            )
        else:
            prompts = zip(instruction_roles, instructions, output_roles, outputs)
        prompts = ["\n".join(i) for i in prompts]
        return {"text": prompts}

    def group_texts(examples, config):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= config.context_length:
            total_length = (
                total_length // config.context_length
            ) * config.context_length
        else:
            total_length = 0
        result = {
            k: [
                t[i : i + config.context_length]
                for i in range(0, total_length, config.context_length)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def tokenize(examples, tokenizer, config, truncation=False):
        output = tokenizer(examples[config.dataset_text_column], truncation=truncation)
        return output

    if train_config.data_processing == "alpaca":
        data = data.map(
            lambda x: alpaca(x, tokenizer, train_config),
            batched=True,
            remove_columns=list(data.features)
            if not train_config.dataset_kwargs["streaming"]
            else train_config.all_columns,
        )
        data = data.map(
            lambda x: tokenize(x, tokenizer, train_config, truncation=True),
            batched=True,
            remove_columns=list(data.features)
            if not train_config.dataset_kwargs["streaming"]
            else train_config.all_columns,
        )
    elif train_config.data_processing == "packing":
        data = data.map(
            lambda x: tokenize(x, tokenizer, train_config),
            batched=True,
            remove_columns=list(data.features)
            if not train_config.dataset_kwargs["streaming"]
            else train_config.all_columns,
        )
        data = data.map(
            lambda x: group_texts(x, train_config),
            batched=True,
        )
    elif train_config.data_processing == "alpaca-packing":
        data = data.map(
            lambda x: alpaca(x, tokenizer, train_config),
            batched=True,
            remove_columns=list(data.features)
            if not train_config.dataset_kwargs["streaming"]
            else train_config.all_columns,
        )
        data = data.map(
            lambda x: tokenize(x, tokenizer, train_config),
            batched=True,
            remove_columns=['text']
            #if not train_config.dataset_kwargs["streaming"]
            #else train_config.all_columns,
        )
        data = data.map(
            lambda x: group_texts(x, train_config),
            batched=True,
        )

    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=train_config.batch_size,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            max_steps=train_config.max_steps,
            num_train_epochs=train_config.num_train_epochs,
            warmup_steps=train_config.warmup_steps,
            learning_rate=train_config.learning_rate,
            bf16=True if train_config.torch_dtype is torch.bfloat16 else False,
            logging_steps=train_config.logging_steps,
            output_dir="outputs",
            optim=train_config.optim,
            report_to="wandb",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    model.config.use_cache = False
    trainer.train()
    if train_config.output_is_repo:
        #if train_config.use_lora:
        #    model.push_to_hub(train_config.output)
        #    return
        #model.push_to_hub(train_config.output)
        #tokenizer.push_to_hub(train_config.output)
        model.save_pretrained(train_config.output.split("/")[-1])
        from huggingface_hub import HfApi
        api.upload_folder(
            folder_path=train_config.output.split("/")[-1],
            path_in_repo=train_config.output.split("/")[-1],
            repo_id="/".join(train_config.output.split("/")[:-1]),
            repo_type="model"
	)
    else:
        #model.save_pretrained(train_config.output)
        #if not train_config.use_lora:
        #    tokenizer.save_pretrained(train_config.output)
        model.save_pretrained(train_config.output)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.json")
args = parser.parse_args()
main(args)
