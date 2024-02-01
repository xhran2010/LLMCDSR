import torch
from transformers import LogitsProcessor
import os
from operator import itemgetter
import pickle
import re
import requests
import json

class StopAfterSpaceIsGenerated(LogitsProcessor):
    def __init__(self, enter_token_id: int, eos_token_id: int, device: str):
        super().__init__()

        self.enter_token_id = enter_token_id
        self.eos_token_id = eos_token_id
        self.device = device

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        forced_eos = torch.full((scores.size(1),), -float("inf")).to(self.device)
        forced_eos[self.eos_token_id] = 0
        
        end_sign = torch.LongTensor([self.enter_token_id, self.enter_token_id, self.enter_token_id]).to(self.device)
        scores[torch.all(input_ids[:, -3:] == end_sign, dim=1)] = forced_eos
        return scores

def sample_by_rank(data, rank):
    local_rank, all_rank = [int(i) for i in rank.split('/')]
    data_len = len(data)
    len_per_rank = int(data_len / (all_rank + 1))
    if local_rank < all_rank:
        rank_data = data[local_rank * len_per_rank:(local_rank + 1) * len_per_rank]
    else:
        rank_data = data[local_rank * len_per_rank:]
    return rank_data, local_rank

def sample_by_subset(A, B, path, is_subsample=False):
    sampled_A = [(0, idx + 1, data) for idx, data in enumerate(A)]
    sampled_B = [(1, idx + 1, data) for idx, data in enumerate(B)]
    
    return sampled_A + sampled_B

def parse_generated_candidates(raw_generation):
    """
    if len(lines) > 1:
        1) -|No. (")A(") (by B) :|- ...
    
    elif len(lines) == 1:
        match strings like "A" and extract A.
    """
    if len(raw_generation.strip().split('\n')) > 1:
        # pattern = r'(?<=\d\. )["“]*(.+?)[”"]*(?=[\(:]| by)|(?<=- )(.+?)(?=:)|(?<=: )(.+?)(?=\[)' # for movie
        pattern = r'\d+\.\s(.*?)\n'
        it = re.findall(pattern, raw_generation)
        extracted_title = list(set(it))
        # extracted_title = [name.strip() for tuple_ in it for name in tuple_ if name] # for movie
        # extracted_title = list(set(extracted_title))
        if len(extracted_title) <= 2:
            return None
        return list(set(extracted_title))
    else:
        return None

def concat_icl(context, texts, source='A'):
    final_prompt = ''
    for example_index, (A_ctx, B_ctx) in enumerate(context, start=1):
        if source == 'A':
            first_ctx = A_ctx
            last_ctx = B_ctx
        else:
            first_ctx = B_ctx
            last_ctx = A_ctx
        final_prompt += f"### Example {example_index}\nInput:"
        final_prompt += str(first_ctx) + '\nOutput:\n'
        final_prompt += '\n'.join(["- " + i for i in last_ctx])
        final_prompt += '\n\n'
    final_prompt += f"\n### Input:\n{str(texts)}"
    return final_prompt