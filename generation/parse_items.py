from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from transformers.generation.utils import GenerationConfig
import torch
from peft import PeftModel
import argparse
import time
import pickle as pkl
from tqdm import tqdm
from ipdb import set_trace
import json
import os
import sys
import random

from generation_utils import (StopAfterSpaceIsGenerated, sample_by_rank, sample_by_subset)

PROMPT = "Extract the names of the items directly, and list them with numbers starting with 1. If there are no items, Don't output anything."

def initialize(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "left"
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tensor_type = torch.bfloat16
    inference_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=tensor_type,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    # inference_model.resize_token_embeddings(len(tokenizer))
    inference_model.generation_config = GenerationConfig.from_pretrained(args.base_model)

    return tokenizer, inference_model


def inference(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt,
        padding=True,
        truncation=True,
        max_length=1999,
        return_tensors='pt',)
    input_ids = inputs['input_ids'].to(device)
    role_token = torch.LongTensor([model.generation_config.user_token_id]).view(1, 1).expand(input_ids.shape[0], -1).to(device)
    input_ids = torch.cat([input_ids, role_token], dim=-1)
    # print(input_ids.shape)
    attention_mask = inputs['attention_mask'].to(device)
    attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1)).to(device)], dim=-1)
    stopper = LogitsProcessorList([StopAfterSpaceIsGenerated(5, 2, device)])
    outputs = model.generate(input_ids=input_ids, 
        attention_mask=attention_mask, 
        eos_token_id=model.generation_config.eos_token_id, 
        max_new_tokens=500,
        remove_invalid_values=True,
        logits_processor=None)
    decoded = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:].detach(), skip_special_tokens=True)
    return decoded

def main(args):
    # list all file names in a dir
    data = []
    data_path = os.path.join("./generation", args.data_name)
    for file_name in os.listdir(data_path):
        with open(os.path.join(data_path, file_name), 'r') as f:
            content = '\n'.join(f.readlines())
            content = '"{}"'.format(content)
            content += '\n\n' + PROMPT
            data.append((file_name, content))
    if args.rank != "all":
        data, local_rank = sample_by_rank(data, args.rank)
    else:
        local_rank = 0

    tokenizer, model = initialize(args)
    if not os.path.exists(os.path.join(args.output_path, args.data_name)):
        os.mkdir(os.path.join(args.output_path, args.data_name))
    for index in tqdm(range(0, len(data), args.batch_size), ncols=0):
        file_names = [i[0] for i in data[index:index + args.batch_size]]
        prompts = [i[1] for i in data[index:index + args.batch_size]]
        decoded = inference(model, tokenizer, prompts, args.device)
        for gen, file_name, prompt in zip(decoded, file_names, prompts):
            with open(os.path.join(args.output_path, args.data_name, f'{file_name}'), 'w') as f:
                # f.write(prompt + '\n\n########\n\n' + gen)
                f.write(gen)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_name', type=str, default="mb-candidates-icl")
    parser.add_argument('--output_path', type=str, default="./parsed_items")
    parser.add_argument('--rank', type=str, default='all')

    args = parser.parse_args()
    main(args)

