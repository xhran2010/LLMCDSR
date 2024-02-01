from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch

from data import RecDataset
from torch.utils.data import DataLoader

class Evaluator(object):
    def __init__(self, dataset, Anum, Bnum, max_len, batch_size, device):
        self.dataset = dataset
        self.Anum = Anum
        self.Bnum = Bnum
        self.maxlen = max_len
        self.device = device
        self.batch_size = batch_size
        self.neg_cands = self._generate_neg()

    def _generate_neg(self):
        negs = []
        print("Generating negative samples for evaluation...")
        for (_, domain, target) in tqdm(self.dataset, disable=True):
            item_idx = [target]
            if domain == 0:
                left, right = 1, self.Anum + 1
            elif domain == 1:
                left, right = self.Anum + 1, self.Anum + self.Bnum + 1
            neg_cands = np.random.choice(list(set(range(left, right)) - set(item_idx)), size=999, replace=False).tolist()
            negs.append(neg_cands)
        return negs
        

    def loss(self, model):
        dataset = {user + 1: i[0] + [i[2]] for user, i in enumerate(self.dataset)}
        eval_dataset = RecDataset(dataset, len(dataset), self.Anum + self.Bnum, self.Anum, maxlen=self.maxlen)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
        loss_sum = 0.
        for step, batch in enumerate(eval_loader):
            u, seq, pos, neg = [x.to(self.device) for x in batch]
            pos[:,:-1] = 0
            loss = model(seq, pos, neg)
            loss_sum += loss.item()
        loss_sum /= len(eval_loader)
        return loss_sum

    def __call__(self, model):

        NDCG_A = defaultdict(float)
        HR_A = defaultdict(float)
        NDCG_B = defaultdict(float)
        HR_B = defaultdict(float)
        valid_user_A = 0.0
        valid_user_B = 0.0
        cutoff_list = [10, 5, 3, 1]

        for ((his, domain, target), neg_cands) in tqdm(zip(self.dataset, self.neg_cands), disable=True):
            seq = np.zeros([self.maxlen], dtype=np.int32)
            idx = self.maxlen - 1
            for i in reversed(his):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            item_idx = [target]
            if domain == 0:
                valid_user_A += 1
            elif domain == 1:
                valid_user_B += 1
            item_idx += neg_cands
            seq = seq.tolist()
            predictions = - model.predict(*[torch.LongTensor(l).to(self.device) for l in [[0], [seq], item_idx]], domain)
            predictions = predictions[0] # - for 1st argsort DESC

            rank = predictions.argsort().argsort()[0].item()

            if domain == 0:
                for cutoff in cutoff_list:
                    if rank < cutoff:
                        NDCG_A[cutoff] += 1 / np.log2(rank + 2)
                        HR_A[cutoff] += 1
            elif domain == 1:
                for cutoff in cutoff_list:
                    if rank < int(cutoff):
                        NDCG_B[cutoff] += 1 / np.log2(rank + 2)
                        HR_B[cutoff] += 1

        for cutoff in cutoff_list:
            NDCG_A[cutoff] /= valid_user_A
            HR_A[cutoff] /= valid_user_A
            NDCG_B[cutoff] /= valid_user_B
            HR_B[cutoff] /= valid_user_B

        return NDCG_A, HR_A, NDCG_B, HR_B
