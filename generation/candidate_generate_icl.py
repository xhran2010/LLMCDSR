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

from generation_utils import (StopAfterSpaceIsGenerated, sample_by_rank, concat_icl, sample_by_subset)


def prepare_data(data_dir, shot, is_overlap=False):
    with open(os.path.join(data_dir, 'item_set_A.pkl'), 'rb') as f:
        id2idx_A = pkl.load(f)
        idx2id_A = {v: k for k, v in id2idx_A.items()}
    with open(os.path.join(data_dir, 'item_set_B.pkl'), 'rb') as f:
        id2idx_B = pkl.load(f)
        idx2id_B = {v: k for k, v in id2idx_B.items()}
    with open(os.path.join(data_dir, 'text_A.pkl'), 'rb') as f:
        id2text_A = pkl.load(f)
        text2id_A = {v: k for k, v in id2text_A.items()}
    with open(os.path.join(data_dir, 'text_B.pkl'), 'rb') as f:
        id2text_B = pkl.load(f)
        text2id_B = {v: k for k, v in id2text_B.items()}
    with open(os.path.join(data_dir, 'num.txt'), 'r') as f:
        num_A, num_B = f.readlines()
        num_A = eval(num_A.strip())
        num_B = eval(num_B.strip())

    A_prompts = []
    B_prompts = []
    with open(os.path.join(data_dir, "candidate_generate_A_icl.txt"), 'r') as f:
        A_prefix = f.read()
    with open(os.path.join(data_dir, "candidate_generate_B_icl.txt"), 'r') as f:
        B_prefix = f.read()
    
    with open(os.path.join(data_dir, 'train_overlap.txt'), 'r') as f:
        ctxs = []
        for l in f:
            _, records = l.strip().split('\t')
            records = records.split(',')
            A_texts = []
            B_texts = []
            for item_idx in records:
                if int(item_idx) < num_A:
                    item_id = idx2id_A[int(item_idx)]
                    text = id2text_A[item_id]
                    A_texts.append(text)
                else:
                    item_id = idx2id_B[int(item_idx) - num_A]
                    text = id2text_B[item_id]
                    B_texts.append(text)
            ctxs.append((A_texts, B_texts))

    A_prompts, B_prompts = [], []
    with open(os.path.join(data_dir, 'train_A.txt'), 'r') as f:
        for l in f:
            _, records = l.strip().split('\t')
            records = records.split(',')
            A_texts = []
            for item_idx in records:
                item_id = idx2id_A[int(item_idx)]
                text = id2text_A[item_id]
                A_texts.append(text)
            context = random.sample(ctxs, k=shot)
            A_prompts.append(A_prefix + concat_icl(context, A_texts, source='A'))
    with open(os.path.join(data_dir, 'train_B.txt'), 'r') as f:
        for l in f:
            _, records = l.strip().split('\t')
            records = records.split(',')
            B_texts = []
            for item_idx in records:
                item_id = idx2id_B[int(item_idx) - num_A]
                text = id2text_B[item_id]
                B_texts.append(text)
            context = random.sample(ctxs, k=shot)
            B_prompts.append(B_prefix + concat_icl(context, B_texts, source='B'))

    return A_prompts, B_prompts


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
    A_data, B_data = prepare_data(args.data_path, args.shot)
    data = sample_by_subset(A_data, B_data, args.data_path) # (domain, user_id, data)
    if args.rank != "all":
        data, local_rank = sample_by_rank(data, args.rank)
    else:
        local_rank = 0

    tokenizer, model = initialize(args)
    if not os.path.exists(os.path.join(args.output_path, args.output_name)):
        os.mkdir(os.path.join(args.output_path, args.output_name))
    for index in tqdm(range(0, len(data), args.batch_size), ncols=0):
        domains = [i[0] for i in data[index:index + args.batch_size]]
        ids = [i[1] for i in data[index:index + args.batch_size]]
        prompts = [i[2] for i in data[index:index + args.batch_size]]
        decoded = inference(model, tokenizer, prompts, args.device)
        for gen, domain, user_id in zip(decoded, domains, ids):
            domain = 'A' if domain == 0 else 'B'
            with open(os.path.join(args.output_path, args.output_name, f'{domain}_{user_id}.txt'), 'w') as f:
                f.write(gen)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--shot', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_path', type=str, default="../data/movie-book")
    parser.add_argument('--output_path', type=str, default="./generation")
    parser.add_argument('--output_name', type=str, default="mb-candidates-icl")
    parser.add_argument('--rank', type=str, default='all')

    args = parser.parse_args()
    main(args)

