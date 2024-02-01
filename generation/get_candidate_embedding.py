import pickle as pkl
from tqdm import tqdm
from torch.utils.data import Dataset
import os
import argparse
from ipdb import set_trace
import sys

from generation_utils import parse_generated_candidates
from get_item_embedding import prepare_model, get_embedding

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--generation_path', type=str, default="./generation/mb-candidates-icl")
    parser.add_argument('--save_path', default='../pretrained_parameters/', type=str)

    args = parser.parse_args()
    return args


class GenerationDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        for file_name in tqdm(os.listdir(data_dir)):
            with open(os.path.join(data_dir, file_name), 'r') as f:
                content = f.read()
                res = parse_generated_candidates(content)
                if res is None:
                    continue
                file_name = file_name[:-4]
                domain, user_id = file_name.split('_')
                domain = 0 if domain == 'A' else 1
                user_id = int(user_id)
                self.data.append((domain, user_id, res))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        domain, user_id, item_title = self.data[index]
        return domain, user_id, item_title


def runner(data, model, batch_size):
    final_embedding_data = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        flatten_titles = []
        record_length = []
        for record in batch:
            record_length.append(len(record[2]))
            flatten_titles.extend(record[2])
        embeddings = get_embedding(flatten_titles, model)
        split_embeddings = []
        last_slice_index = 0
        for slice_len in record_length:
            sliced_embedding = embeddings[last_slice_index:last_slice_index + slice_len]
            split_embeddings.append(sliced_embedding)
            last_slice_index += slice_len
        for index, record in enumerate(batch):
            final_embedding_data.append((record[0], record[1], split_embeddings[index]))
    return final_embedding_data

if __name__ == "__main__":
    args = get_args()
    generation_dataset = GenerationDataset(data_dir=args.generation_path)
    model = prepare_model(args.model_path, args.device)
    generation_with_embedding = runner(generation_dataset.data, model, args.batch_size)
    with open(os.path.join(args.save_path, "{}_generation_jina.pkl".format(args.task)), 'wb') as f:
        pkl.dump(generation_with_embedding, f)