import pickle
import numpy as np
import argparse
from transformers import AutoModel
from ipdb import set_trace
from tqdm import tqdm
from os.path import join as path_join

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--domain', default='A', type=str)
    parser.add_argument('--exp_name', default='default', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--save_path', default='../pretrained_parameters/', type=str)

    args = parser.parse_args()
    if args.device == -1:
        args.device = 'cpu'
    else:
        args.device = "cuda:{}".format(args.device)
    return args

def get_data(task, domain):
    data_dict = {}
    with open('../data/{}/text_{}.pkl'.format(task, domain), 'rb') as f:
        text_descriptions = pickle.load(f)
    with open('../data/{}/item_set_{}.pkl'.format(task, domain), 'rb') as f:
        id_index_map = pickle.load(f)
    for item_id, index in id_index_map.items():
        text = text_descriptions[item_id]
        data_dict[index] = text
    data = []
    for i in range(len(data_dict)):
        data.append(data_dict[i])
    return data

def prepare_model(model_path, device):
    model = AutoModel.from_pretrained(model_path, 
                                      trust_remote_code=True, 
                                      device_map={"": device},
                                      local_files_only=True)
    return model

def get_embedding(text_batch, model):
    embeddings = model.encode(text_batch)
    return embeddings

def runner(data, model, batch_size):
    final_embeddings_list = []
    for i in tqdm(range(0, len(data), batch_size)):
        # print("{}-{}/{}".format(i, i + batch_size, len(data)))
        batch = data[i:i + batch_size]
        embeddings = get_embedding(batch, model)
        final_embeddings_list.append(embeddings)
    final_embedding_mat = np.concatenate(final_embeddings_list, axis=0)
    print(final_embedding_mat.shape)
    return final_embedding_mat

def save_mat(mat, save_path, task, domain):
    np.save(path_join(save_path, "{}_{}_item_jina.npy".format(task, domain)), mat)
    
def main(args):
    data = get_data(args.task, args.domain)
    model = prepare_model(args.model_path, args.device)
    mat = runner(data, model, args.batch_size)
    save_mat(mat, args.save_path, args.task, args.domain)
    
if __name__ == "__main__":
    args = get_args()
    main(args)