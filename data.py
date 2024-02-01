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


class RecDataset(Dataset):
    def __init__(self, User, usernum, itemnum, Anum, aug_cands=None, maxlen=50):
        self.User = User
        self.usernum = usernum
        self.itemnum = itemnum
        self.Anum = Anum
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
            if int(nxt) <= self.Anum:
                neg[idx] = random_neq(1, self.Anum + 1, ts)
            else:
                neg[idx] = random_neq(self.Anum + 1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return user, seq, pos, neg

    def __len__(self):
        return len(self.User)


class AugDataset(Dataset):
    def __init__(self, User):
        self.User = User

    def __getitem__(self, user):
        return self.User[user]

    def __len__(self):
        return len(self.User)
    
    @staticmethod
    def collate_fn(batch):
        domain, source_seq, cand = zip(*batch)
        domain = torch.LongTensor(domain)

        max_length = max(t.size(0) for t in cand)
        padded_cand = []
        for t in cand:
            pad_size = max_length - t.size(0)
            # Pad at the end along the L dimension
            padded_tensor = F.pad(t, (0, 0, 0, pad_size))
            padded_cand.append(padded_tensor)
        padded_cand = torch.stack(padded_cand, dim=0)
        return domain, source_seq, padded_cand
