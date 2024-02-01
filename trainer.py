from utils import EarlyStopping
from os.path import join as path_join
import torch
import logging
import numpy as np
import wandb
from torch.utils.data import DataLoader
import math
from itertools import cycle
from ipdb import set_trace

from model import TestModel
from copy import deepcopy

import higher

def train_meta_step(shadow_model, meta_model, meta_opt, 
                    origin_instance, noisy_instance, meta_instance, 
                    shadow_lr, l2_emb, meta_step=1, shadow_opt_state=None):
    # test_model = TestModel().to('cuda:0')
    aug_domain, aug_rep, aug_his = noisy_instance
    _, meta_seq, meta_pos, meta_neg = meta_instance
    ori_seq, ori_pos, ori_neg = origin_instance
    final_opt_state = None
    meta_loss_sum = 0.
    for step in range(meta_step):
        noisy_seq, noisy_pos, noisy_neg, noisy_seq_emb = meta_model(aug_domain, aug_rep, aug_his, shadow_model)
        ####### shadow step #######
        copy_shadow_model = deepcopy(shadow_model)
        copy_shadow_model.train()
        shadow_opt = torch.optim.Adam(copy_shadow_model.parameters(), lr=shadow_lr, betas=(0.9, 0.98), weight_decay=l2_emb)
        if shadow_opt_state is not None:
            shadow_opt.load_state_dict(shadow_opt_state)
        with higher.innerloop_ctx(copy_shadow_model, shadow_opt) as (fshadow_model, fshadow_opt):
            shadow_loss = fshadow_model.shadow_forward(
                ori_seqs=(ori_seq, ori_pos, ori_neg),
                aug_seqs=(noisy_seq, noisy_pos, noisy_neg),
                aug_embs=(noisy_seq_emb,),
            )
            fshadow_opt.step(shadow_loss)
            ####### meta step #######
            meta_loss = fshadow_model(meta_seq, meta_pos, meta_neg)
        meta_opt.zero_grad()
        meta_loss.backward()
        meta_opt.step()
        meta_loss_sum += meta_loss.item()
        if step == meta_step - 1:
            final_opt_state = shadow_opt.state_dict()
    meta_loss_sum /= meta_step
    
    return final_opt_state, meta_loss_sum


def train(model, rec_dataset, aug_dataset, meta_dataset, evaluators, args, pooling_model):
    early_stopping_A = EarlyStopping(
        checkpoint_path=path_join(args.save_dir, "best_A.pt"),
        patience=args.patience,
        label='A'
    )
    early_stopping_B = EarlyStopping(
        checkpoint_path=path_join(args.save_dir, "best_B.pt"),
        patience=args.patience,
        label='B'
    )
    dataloader = DataLoader(rec_dataset, batch_size=args.batch_size, shuffle=True)
    num_batch = len(dataloader)
    aug_iter_func = lambda: iter(DataLoader(aug_dataset, batch_size=args.aug_batch_size, shuffle=True, collate_fn=aug_dataset.collate_fn, num_workers=1))
    aug_iter = aug_iter_func()

    meta_iter_func = lambda: iter(DataLoader(meta_dataset, batch_size=args.meta_batch_size, shuffle=True))
    meta_iter = meta_iter_func()
    shadow_opt_state = None
    meta_opt = torch.optim.Adam(pooling_model.parameters(), lr=args.meta_lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)
    
    model.train()  # enable model training

    val_eval, test_eval = evaluators
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)

    for epoch in range(1, args.num_epochs + 1):
        if args.inference: break
        loss_sum = 0.
        meta_loss_sum = 0.
        total_meta_steps = num_batch // args.meta_interval
        for step, batch in enumerate(dataloader):
            u, seq, pos, neg = [x.to(args.device) for x in batch]
            try:
                aug_domain, aug_his, aug_rep = next(aug_iter)
            except StopIteration:
                aug_iter = aug_iter_func()
                aug_domain, aug_his, aug_rep = next(aug_iter)
            aug_domain = aug_domain.to(args.device)
            aug_rep = aug_rep.to(args.device)
            if (step + 1) % args.meta_interval == 0:
                shadow_model = deepcopy(model)
                shadow_model.train()
                try:
                    meta_instance = next(meta_iter)
                except StopIteration:
                    meta_iter = meta_iter_func()
                    meta_instance = next(meta_iter)
                meta_instance = [x.to(args.device) for x in meta_instance]
                shadow_opt_state, meta_loss = train_meta_step(shadow_model, pooling_model, meta_opt,
                                                origin_instance=(seq, pos, neg),
                                                noisy_instance=(aug_domain, aug_rep, aug_his),
                                                meta_instance=meta_instance,
                                                shadow_lr=args.shadow_lr,
                                                l2_emb=args.l2_emb,
                                                meta_step=args.meta_step,
                                                shadow_opt_state=shadow_opt_state)
                meta_loss_sum += meta_loss

            aug_seq, aug_pos, aug_neg, _ = pooling_model(aug_domain, aug_rep, aug_his, model)
            seq = torch.cat([seq, aug_seq], dim=0)
            pos = torch.cat([pos, aug_pos], dim=0)
            neg = torch.cat([neg, aug_neg], dim=0)
            
            loss = model(seq, pos, neg)
            loss_sum += loss.item()

            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()
            if args.log_step != -1 and step % args.log_step == 0:
                logging.info(f"Epoch {epoch:02d} | iteration {step:04d}/{num_batch:04d} | Loss {loss.item():.3f}")

        loss_sum /= num_batch
        meta_loss_sum /= total_meta_steps
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "overall loss": loss_sum,
                "meta loss": meta_loss_sum,
            })
        logging.info(
            f"Epoch {epoch:02d} | Loss {loss_sum:.3f} | Meta Loss {meta_loss_sum:.3f}")

        if epoch % args.eval_epoch == 0:
            model.eval()
            logging.info('Evaluating')
            val_loss = val_eval.loss(model)
            test_loss = test_eval.loss(model)
            val_NDCG_A, val_HR_A, val_NDCG_B, val_HR_B = val_eval(model)
            test_NDCG_A, test_HR_A, test_NDCG_B, test_HR_B = test_eval(model)
            early_stopping_A(val_HR_A[10], model, pooling_model)
            early_stopping_B(val_HR_B[10], model, pooling_model)

            if early_stopping_A.early_stop and early_stopping_B.early_stop:
                break

            val_key_map = {
                'A':{'NDCG': val_NDCG_A,'HR': val_HR_A},
                'B':{'NDCG': val_NDCG_B,'HR': val_HR_B},
            }
            test_key_map = {
                'A': {'NDCG': test_NDCG_A, 'HR': test_HR_A},
                'B': {'NDCG': test_NDCG_B, 'HR': test_HR_B},
            }
            wandb_dict = {'epoch': epoch, 'val_loss': val_loss, 'test_loss': test_loss}
            for domain in ['A', 'B']:
                for metric in ['NDCG', 'HR']:
                    for k in [1, 3, 5, 10]:
                        wandb_dict.update({f"val_{domain}_{metric}@{k}": val_key_map[domain][metric][k]})
            for domain in ['A', 'B']:
                for metric in ['NDCG', 'HR']:
                    for k in [1, 3, 5, 10]:
                        wandb_dict.update({f"test_{domain}_{metric}@{k}": test_key_map[domain][metric][k]})
            if args.wandb: wandb.log(wandb_dict)
            log_str = \
                f'Epoch {epoch:02d} | Valid | A@1  | NDCG {val_NDCG_A[1]:.4f} | HR {val_HR_A[1]:.4f}\n' + \
                f'Epoch {epoch:02d} | Valid | A@3  | NDCG {val_NDCG_A[3]:.4f} | HR {val_HR_A[3]:.4f}\n' + \
                f'Epoch {epoch:02d} | Valid | A@5  | NDCG {val_NDCG_A[5]:.4f} | HR {val_HR_A[5]:.4f}\n' + \
                f'Epoch {epoch:02d} | Valid | A@10 | NDCG {val_NDCG_A[10]:.4f} | HR {val_HR_A[10]:.4f}\n' + \
                f'Epoch {epoch:02d} | Valid | B@1  | NDCG {val_NDCG_B[1]:.4f} | HR {val_HR_B[1]:.4f}\n' + \
                f'Epoch {epoch:02d} | Valid | B@3  | NDCG {val_NDCG_B[3]:.4f} | HR {val_HR_B[3]:.4f}\n' + \
                f'Epoch {epoch:02d} | Valid | B@5  | NDCG {val_NDCG_B[5]:.4f} | HR {val_HR_B[5]:.4f}\n' + \
                f'Epoch {epoch:02d} | Valid | B@10 | NDCG {val_NDCG_B[10]:.4f} | HR {val_HR_B[10]:.4f}\n' + \
                f'Epoch {epoch:02d} | Test  | A@1  | NDCG {test_NDCG_A[1]:.4f} | HR {test_HR_A[1]:.4f}\n' + \
                f'Epoch {epoch:02d} | Test  | A@3  | NDCG {test_NDCG_A[3]:.4f} | HR {test_HR_A[3]:.4f}\n' + \
                f'Epoch {epoch:02d} | Test  | A@5  | NDCG {test_NDCG_A[5]:.4f} | HR {test_HR_A[5]:.4f}\n' + \
                f'Epoch {epoch:02d} | Test  | A@10 | NDCG {test_NDCG_A[10]:.4f} | HR {test_HR_A[10]:.4f}\n' + \
                f'Epoch {epoch:02d} | Test  | B@1  | NDCG {test_NDCG_B[1]:.4f} | HR {test_HR_B[1]:.4f}\n' + \
                f'Epoch {epoch:02d} | Test  | B@3  | NDCG {test_NDCG_B[3]:.4f} | HR {test_HR_B[3]:.4f}\n' + \
                f'Epoch {epoch:02d} | Test  | B@5  | NDCG {test_NDCG_B[5]:.4f} | HR {test_HR_B[5]:.4f}\n' + \
                f'Epoch {epoch:02d} | Test  | B@10 | NDCG {test_NDCG_B[10]:.4f} | HR {test_HR_B[10]:.4f}\n' + \
                "=================================================="

            logging.info(log_str)
            model.train()

    return early_stopping_A, early_stopping_B