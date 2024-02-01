import numpy as np
import torch
from utils import NCELoss
from collections import defaultdict
import copy
import random
from data import random_neq
from ipdb import set_trace

from torch import nn
import torch.nn.init as init

class ProjectionModule(nn.Module):
    def __init__(self, text_weight, proj_in_size, proj_out_size, temperature, device):
        super(ProjectionModule, self).__init__()
        self.text_weight = text_weight
        self.projection = nn.Linear(proj_in_size, proj_out_size, bias=False)
        self.cl_loss = NCELoss(temperature=temperature, device=device)
        self._init_weights()
    
    def _init_weights(self):
        init.xavier_normal_(self.projection.weight)

    def forward(self, items, cf_embeddings):
        # remove padding idx in items, i.e., 0
        items = items[items != 0]
        items = items - 1
        text_embedding = self.text_weight[items]
        proj_cf_embedding = self.projection(text_embedding)
        cf_embedding = cf_embeddings(items)
        loss = self.cl_loss(proj_cf_embedding, cf_embedding)
        return loss

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

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, text_weight, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.cl_weight = args.cl_weight

        self.proj_model = ProjectionModule(
                text_weight=text_weight,
                proj_in_size=768,
                proj_out_size=args.hidden_units,
                temperature=args.temperature,
                device=args.device
            )

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
        rec_loss = self.bce(pos_logits[indices], pos_labels[indices])
        rec_loss += self.bce(neg_logits[indices], neg_labels[indices])

        involved_items = torch.unique(torch.cat([log_seqs, pos_seqs, neg_seqs], dim=0))
        cl_loss = self.proj_model(involved_items, self.item_emb)

        loss = rec_loss + self.cl_weight * cl_loss

        return loss, rec_loss, cl_loss

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self.item_emb(item_indices) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        # preds = self.pos_sigmoid(logits) # rank same item list for different users
        return logits # preds # (U, I)

