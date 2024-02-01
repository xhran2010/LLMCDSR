import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
import pickle
from tqdm import tqdm
import torch.nn as nn
import os
import logging
from ipdb import set_trace


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, label="", patience=20, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_model = None
        self.label = label
        if self.label != "":
            self.label += " "

    def compare(self, score):
        if score > self.best_score + self.delta:
            return False
        return True

    def __call__(self, score, model, side_model=None):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = 0
            self.save_checkpoint(score, model)
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.early_stop:
            logging.info(f"{self.label}has early stopped.")
        elif self.compare(score):
            self.counter += 1
            logging.info(f'{self.label}earlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            logging.info(f'{self.label}earlyStopping counter reset!')
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            if side_model is not None:
                self.save_checkpoint(score, side_model, side=True)
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model, side=False):
        '''Saves model when validation loss decrease.'''
        if not side:
            if self.verbose:
                # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
                logging.info(f'Validation score increased.  Saving model ...')
            torch.save(model.state_dict(), self.checkpoint_path)
            self.score_min = score
        else:
            torch.save(model.state_dict(), self.checkpoint_path + '.side')


class NCELoss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """
    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)
        
    # #modified based on impl:
    # https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two):
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss


def load_eval(dataset, max_len=50):
    user_valid, user_test = [], []
    with open(f'./data/{dataset}/num.txt', 'r') as f:
        Anum = int(f.readline().strip())
        Bnum = int(f.readline().strip())
    with open(f'data/{dataset}/valid.txt', 'r') as f:
        for line in f:
            u, is_ = line.rstrip().split('\t')
            u = int(u)
            is_ = [int(i) + 1 for i in is_.split(',')]
            target = is_[-1]
            target_domain = 0 if target <= Anum else 1
            user_valid.append((is_[:-1], target_domain, target))
    with open(f'data/{dataset}/test.txt', 'r') as f:
        for line in f:
            u, is_ = line.rstrip().split('\t')
            u = int(u)
            is_ = [int(i) + 1 for i in is_.split(',')]
            target = is_[-1]
            target_domain = 0 if target <= Anum else 1
            user_test.append((is_[:-1], target_domain, target))
    return user_valid, user_test, Anum, Bnum


# train/val/test data generation
def data_partition(fname, overlap_ratio=1.0):
    user_train = {}
    user_single = {}
    user_overlap = {}
    # assume user/item index starting from 1
    with open('data/%s/num.txt' % fname, 'r') as f:
        itemnum_A = int(f.readline().strip())
        itemnum_B = int(f.readline().strip())
    itemnum = itemnum_A + itemnum_B

    with open('data/%s/instance_num.txt' % fname, 'r') as f:
        instance_num_A = int(f.readline().strip())
        instance_num_B = int(f.readline().strip())
        instance_num_overlap = int(f.readline().strip())

    with open('data/%s/train_A.txt' % fname, 'r') as f:
        A_counter = 0
        for line in f:
            u, is_ = line.rstrip().split('\t')
            u = int(u)
            is_ = [int(i) + 1 for i in is_.split(',')]
            user_single[u] = is_
            user_train[u] = is_
            A_counter += 1

    current_users = len(user_train)
    single_users = len(user_single)
    with open('data/%s/train_B.txt' % fname, 'r') as f:
        B_counter = 0
        for line in f:
            u, is_ = line.rstrip().split('\t')
            u = int(u)
            is_ = [int(i) + 1 for i in is_.split(',')]
            user_single[u + single_users] = is_
            user_train[u + current_users] = is_
            B_counter += 1
    
    current_users = len(user_train)
    with open('data/%s/train_overlap.txt' % fname, 'r') as f:
        overlap_counter = 0
        for line in f:
            u, is_ = line.rstrip().split('\t')
            u = int(u)
            is_ = [int(i) + 1 for i in is_.split(',')]
            if overlap_counter < instance_num_overlap * overlap_ratio:
                user_train[u + current_users] = is_
            user_overlap[u] = is_
            overlap_counter += 1
    usernum = len(user_train)
    return user_train, user_single, user_overlap, usernum, instance_num_A, itemnum


def load_llm_generation(dataset, train_users, usernum_A, backbone):
    if backbone == 'baichuan':
        backbone = ''
    else:
        backbone = '_' + backbone
    with open("./pretrained_parameters/{}{}_generation_jina.pkl".format(dataset, backbone), 'rb') as f:
        generation_data = pickle.load(f)
        res = []
        for record in generation_data:
            domain, user_id, embed = record
            embed = torch.from_numpy(embed)
            if domain == 1: user_id += usernum_A
            source_seq = train_users[user_id]
            res.append((domain, source_seq, embed))
    return res

def load_item_embedding(dataset, device):
    key_ = 'item_emb.weight'
    A = torch.load("./pretrained_parameters/{}_projection_A.pt".format(dataset), map_location={"cuda:1": device})[key_].data.cpu().to(device)
    B = torch.load("./pretrained_parameters/{}_projection_B.pt".format(dataset), map_location={"cuda:7": device})[key_].data.cpu().to(device)
    A = A[1:,:]
    B = B[1:,:]
    return A, B

def load_projection(dataset, device):
    key_ = 'proj_model.projection.weight'
    A = torch.load("./pretrained_parameters/{}_projection_A.pt".format(dataset), map_location={"cuda:1": device})[key_].data.cpu().to(device)
    B = torch.load("./pretrained_parameters/{}_projection_B.pt".format(dataset), map_location={"cuda:7": device})[key_].data.cpu().to(device)
    return A, B