import torch
import numpy as np
import random
from torch.utils.data import Dataset
from tqdm import tqdm
import copy
from collections import defaultdict
import torch.nn.functional as F

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

import numpy as np
import random
import scipy.sparse as sp
import torch.utils.data as data
import torch
from os.path import join as path_join


def load_text_weight(task, domain, device):
    weight = np.load(path_join('../pretrained_parameters', '{}_{}_item_jina.npy'.format(task, domain)))
    return torch.from_numpy(weight).to(device)


class SeqPTDataset:
    def __init__(self, args, domain='A'):
        with open(f'../data/{args.dataset}/num.txt', 'r') as f:
            Anum = int(f.readline().strip())
            Bnum = int(f.readline().strip())
        usernum = 0
        user_train, user_valid = {}, {}
        with open('../data/{}/train_{}.txt'.format(args.dataset, domain), 'r') as f:
            for line in f:
                u, is_ = line.rstrip().split('\t')
                u = int(u)
                if domain == 'A':
                    is_ = [int(i) + 1 for i in is_.split(',')]
                elif domain == 'B':
                    is_ = [int(i) + 1 - Anum for i in is_.split(',')]
                user_train[u], user_valid[u] = self._train_valid_split(is_)
                usernum = max(u, usernum)

        current_users = usernum
        with open('../data/{}/train_overlap.txt'.format(args.dataset), 'r') as f:
            for line in f:
                u, is_ = line.rstrip().split('\t')
                u = int(u) + current_users
                is_ = [int(i) + 1 for i in is_.split(',')]
                single_is = []
                for item in is_:
                    if domain == 'A':
                        if item <= Anum:
                            single_is.append(item)
                    elif domain == 'B':
                        if item > Anum:
                            single_is.append(item - Anum)
                user_train[u], user_valid[u] = self._train_valid_split(single_is)
                usernum = max(u, usernum)
        
        itemnum = Anum if domain == 'A' else Bnum
        self.user_train = user_train
        self.user_valid = user_valid
        self.usernum = usernum
        self.itemnum = itemnum

    def _train_valid_split(self, items):
        train_items = items[:-1]
        val_items = (items[:-1], items[-1])
        return train_items, val_items


class RecDataset(Dataset):
    def __init__(self, User, usernum, itemnum, aug_cands=None, maxlen=50):
        self.User = User
        self.usernum = usernum
        self.itemnum = itemnum
        self.aug_cands = aug_cands
        self.maxlen = maxlen

    def __getitem__(self, user):
        user += 1
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        nxt = self.User[user][-1]
        idx = self.maxlen - 1

        ts = set(self.User[user])

        for i in reversed(self.User[user][:-1]):
            seq[idx] = int(i)
            pos[idx] = int(nxt)
            neg[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return user, seq, pos, neg

    def __len__(self):
        return len(self.User)
