from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch

class Evaluator(object):
    def __init__(self, dataset, itemnum, max_len, device, batch_size=128):
        self.dataset = dataset
        self.itemnum = itemnum
        self.maxlen = max_len
        self.device = device
        self.batch_size = batch_size
        self.all_neg_pool = np.zeros((len(self.dataset), self.itemnum - 1), dtype=np.int64)
        print("Generating negative pool...")
        for row_idx, (his, target) in tqdm(enumerate(self.dataset), disable=True, total=len(self.dataset)):
            item_idx = [target]
            pool = list(set(range(1, self.itemnum + 1)) - set(item_idx))
            self.all_neg_pool[row_idx] = pool

    def __call__(self, model):

        NDCG = defaultdict(float)
        HR = defaultdict(float)
        valid_user = 0.0
        cutoff_list = [10, 5, 3]

        all_his, all_cand = [], []

        neg_index = np.random.randint(0, self.all_neg_pool.shape[1], size=(self.all_neg_pool.shape[0], 999))
        # select all_neg_pool with neg_index
        all_neg = self.all_neg_pool[np.arange(self.all_neg_pool.shape[0])[:, None], neg_index]

        for user_id, (his, target) in tqdm(enumerate(self.dataset), disable=True):
            seq = np.zeros([self.maxlen], dtype=np.int32)
            idx = self.maxlen - 1
            for i in reversed(his):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            item_idx = [target]
            valid_user += 1
            # neg_cands = np.random.choice(list(set(range(1, self.itemnum + 1)) - set(item_idx)), size=999, replace=False).tolist()
            neg_cands = all_neg[user_id].tolist()
            item_idx += neg_cands
            seq = seq.tolist()
            all_his.append(seq)
            all_cand.append(item_idx)
        
        for i in tqdm(range(0, len(all_his), self.batch_size), disable=True):
            seq = all_his[i:i+self.batch_size]
            item_idx = all_cand[i:i+self.batch_size]
            seq = torch.LongTensor(seq).to(self.device)
            item_idx = torch.LongTensor(item_idx).to(self.device)
            user = torch.arange(seq.shape[0]).to(self.device)
            predictions = -model.predict(user, seq, item_idx)
            # predictions = predictions[0] # - for 1st argsort DESC

            rank_list = predictions.argsort(dim=-1).argsort(dim=-1)[:,0].tolist()
            for rank in rank_list:
                for cutoff in cutoff_list:
                    if rank < cutoff:
                        NDCG[cutoff] += 1 / np.log2(rank + 2)
                        HR[cutoff] += 1

        for cutoff in cutoff_list:
            NDCG[cutoff] /= valid_user
            HR[cutoff] /= valid_user

        return NDCG, HR
