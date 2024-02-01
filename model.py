import numpy as np
import torch
from utils import NCELoss
from collections import defaultdict
import copy
import random
from data import random_neq
from ipdb import set_trace

class Recaller(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gen_embedding, item_embedding, k, data_embedding, Anum, is_B=False):
        sim_dot = torch.matmul(gen_embedding, item_embedding.transpose(0,1))
        # sim_dot = torch.cdist(gen_embedding, item_embedding, p=2)
        divider = torch.norm(gen_embedding, dim=-1, keepdim=True) * torch.norm(item_embedding.transpose(0,1), dim=0, keepdim=True)
        sim_score = sim_dot / divider
        # sim_score = sim_dot
        sim_rank = torch.topk(sim_score, k=k).indices + 1
        if is_B:
            sim_rank += Anum
        
        recalled_embedding = data_embedding.index_select(0, sim_rank.view(-1)).view(sim_rank.shape[0], sim_rank.shape[1], -1)

        return sim_rank, recalled_embedding
    
    @staticmethod
    def backward(ctx, *grad_output):
        _, emb_grad = grad_output
        return torch.mean(emb_grad, dim=1), None, None, None, None, None
    

class TestModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.test_emb = torch.nn.Embedding(1, 50)
    
    def forward(self):
        pass


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class Selector(torch.nn.Module):
    def __init__(self, hidden_units, model_hidden, k, device, itemnum, Anum, projection=None, maxlen=50, item_embeddings=None):
        super(Selector, self).__init__()
        self.itemnum = itemnum
        self.Anum = Anum
        self.maxlen = maxlen
        self.data = None
        self.device = device
        self.A_item_emb, self.B_item_emb = item_embeddings
        if type(self.A_item_emb) is not torch.Tensor:
            self.A_item_emb = torch.from_numpy(self.A_item_emb).to(self.device)
            self.B_item_emb = torch.from_numpy(self.B_item_emb).to(self.device)
        self.k = k
        self.projection = projection

        # self.recaller = RecallItemEmbeddings.apply
        self.recaller = Recaller.apply

        # self.pooler = torch.nn.Linear(hidden_units, 1, bias=False)
        self.pooler_A = torch.nn.Linear(model_hidden, 1, bias=False)
        self.pooler_B = torch.nn.Linear(model_hidden, 1, bias=False)

    def forward(self, domains, gen_embedding, his_seqs, rec_model):
        # gen_embedding = self._pooling(gen_embedding)
        A_gen_embedding = torch.matmul(gen_embedding, self.projection[0].transpose(0,1))
        B_gen_embedding = torch.matmul(gen_embedding, self.projection[1].transpose(0,1))

        A_item_emb = self.A_item_emb
        B_item_emb = self.B_item_emb
        A_gen_embedding = self._pooling(A_gen_embedding, 's')
        B_gen_embedding = self._pooling(B_gen_embedding, 't')

        A_recalled, A_recalled_emb = self.recaller(A_gen_embedding, A_item_emb, self.k, rec_model.item_emb.weight, self.Anum)
        B_recalled, B_recalled_emb = self.recaller(B_gen_embedding, B_item_emb, self.k, rec_model.item_emb.weight, self.Anum, True)
        recalled_emb = torch.where(domains.view(-1, 1, 1).expand(-1, self.k, B_recalled_emb.shape[-1]) == 0, B_recalled_emb, A_recalled_emb)

        recalled = torch.where(domains.unsqueeze(1).expand(-1, self.k) == 0, B_recalled, A_recalled)
        batch_seq, batch_pos, batch_neg = [], [], []
        for his, cand in zip(his_seqs, recalled):
            aug_seq, fake_flag = self._random_insert(his, cand)
            seq, pos, neg = self._getitem(aug_seq, fake_flag)
            batch_seq.append(torch.from_numpy(seq))
            batch_pos.append(torch.from_numpy(pos))
            batch_neg.append(torch.from_numpy(neg))
        batch_seq = torch.stack(batch_seq, dim=0).to(self.device)
        batch_pos = torch.stack(batch_pos, dim=0).to(self.device)
        batch_neg = torch.stack(batch_neg, dim=0).to(self.device)
        batch_seq_emb = rec_model.item_emb(batch_seq)

        batch_seq_ext = batch_seq.unsqueeze(-1).expand(-1, -1, self.k)
        recalled_ext = recalled.unsqueeze(1).expand(-1, self.maxlen, -1)
        seq_insert_point = (recalled_ext == batch_seq_ext).nonzero()

        batch_seq_emb[seq_insert_point[:,0], seq_insert_point[:,1], :] = recalled_emb[seq_insert_point[:,0], seq_insert_point[:,2], :]

        return batch_seq, batch_pos, batch_neg, batch_seq_emb

    def _pooling(self, x, domain='s'):
        length_mask = torch.where(torch.sum(x, dim=-1) == 0, 0, 1)
        length = torch.sum(length_mask, dim=-1)
        if domain == 's':
            weight = self.pooler_A(x)
        else:
            weight = self.pooler_B(x) 
        weight = torch.where(length_mask.unsqueeze(2) == 1, weight, -1e9)
        return torch.sum(torch.softmax(weight, dim=1) * x, dim=1)

    def _random_insert(self, sequence, aug_cands):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        # insert_nums = max(int(self.insert_rate*len(copied_sequence)), 1)
        aug_cands = aug_cands.tolist()
        insert_idx = defaultdict(list)
        for i in aug_cands:
            idx = random.choice(list(range(len(copied_sequence))))
            insert_idx[idx].append(i)
        
        inserted_sequence = []
        fake_flag = []
        for index, item in enumerate(copied_sequence):
            inserted_sequence += insert_idx[index]
            inserted_sequence += [item]

            fake_flag += [1] * len(insert_idx[index])
            fake_flag += [0]

        return inserted_sequence, fake_flag
    
    def _getitem(self, aug_data, fake_flag):
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        pos_idx = len(aug_data) - 1
        nxt = aug_data[pos_idx]
        idx = self.maxlen - 1

        ts = set(aug_data)

        for i in reversed(aug_data[:-1]):
            seq[idx] = int(i)
            if fake_flag[pos_idx] == 0:
                pos[idx] = int(nxt)
                if int(nxt) <= self.Anum:
                    neg[idx] = random_neq(1, self.Anum + 1, ts)
                else:
                    neg[idx] = random_neq(self.Anum + 1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            pos_idx -= 1
            if idx == -1: break

        return seq, pos, neg


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, Anum, Bnum, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.Anum = Anum
        self.Bnum = Bnum
        self.item_num = Anum + Bnum
        self.dev = args.device

        self.cl_criterion = NCELoss(args.temperature, args.device)

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
        
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def log2feats(self, log_seqs, log_seqs_emb=None):
        if log_seqs_emb is None:
            seqs = self.item_emb(log_seqs)
        else:
            seqs = log_seqs_emb
        timeline_mask = log_seqs == 0

        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats
    
    def shadow_forward(self, ori_seqs, aug_seqs, aug_embs):
        ori_log_seqs, ori_pos_seqs, ori_neg_seqs = ori_seqs
        aug_log_seqs, aug_pos_seqs, aug_neg_seqs = aug_seqs
        aug_log_seqs_emb, = aug_embs

        log_seqs = torch.cat([ori_log_seqs, aug_log_seqs], dim=0)
        pos_seqs = torch.cat([ori_pos_seqs, aug_pos_seqs], dim=0)
        neg_seqs = torch.cat([ori_neg_seqs, aug_neg_seqs], dim=0)

        ori_log_seqs_emb = self.item_emb(ori_log_seqs)
        log_seqs_emb = torch.cat([ori_log_seqs_emb, aug_log_seqs_emb], dim=0)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        log_feats = self.log2feats(log_seqs, log_seqs_emb) # user_ids hasn't been used yet

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), \
            torch.zeros(neg_logits.shape, device=self.dev)

        indices = torch.where(pos_seqs != 0)
        loss = self.bce(pos_logits[indices], pos_labels[indices])
        loss += self.bce(neg_logits[indices], neg_labels[indices])

        return loss

    def forward(self, log_seqs, pos_seqs, neg_seqs, test_weight=None): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        if test_weight is not None:
            pos_embs += test_weight.unsqueeze(1)
            neg_embs += test_weight.unsqueeze(1)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), \
            torch.zeros(neg_logits.shape, device=self.dev)

        indices = torch.where(pos_seqs != 0)
        loss = self.bce(pos_logits[indices], pos_labels[indices])
        loss += self.bce(neg_logits[indices], neg_labels[indices])
        
        return loss

    def predict(self, user_ids, log_seqs, item_indices, domain): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self.item_emb(item_indices) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        # preds = self.pos_sigmoid(logits) # rank same item list for different users
        return logits # preds # (U, I)
