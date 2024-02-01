from utils import EarlyStopping
from os.path import join as path_join
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from ipdb import set_trace


def train(model, rec_dataset, val_eval, args):
    early_stopping = EarlyStopping(
        checkpoint_path=path_join(args.save_dir, "best.pt"),
        patience=args.patience)
    dataloader = DataLoader(rec_dataset, batch_size=args.batch_size, shuffle=True)
    num_batch = len(dataloader)

    model.train()  # enable model training
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)

    for epoch in range(1, args.num_epochs + 1):
        if args.inference: break
        loss_sum, rec_loss_sum, cl_loss_sum = 0., 0., 0.
        for step, batch in enumerate(dataloader):
            u, seq, pos, neg = [x.to(args.device) for x in batch]
            loss, rec_loss, cl_loss = model(seq, pos, neg)
            loss_sum += loss.item()
            rec_loss_sum += rec_loss.item()
            cl_loss_sum += cl_loss.item()

            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()
            if args.log_step != -1 and step % args.log_step == 0:
                logging.info(f"Epoch {epoch:02d} | iteration {step:04d}/{num_batch:04d} | Loss {loss.item():.3f}")

        loss_sum /= num_batch
        rec_loss_sum /= num_batch
        cl_loss_sum /= num_batch
        logging.info(
            f"Epoch {epoch:02d} | Loss {loss_sum:.3f} | Rec Loss {rec_loss_sum:.3f} | Ctr Loss {cl_loss_sum:.3f} ")

        if epoch % args.eval_epoch == 0:
            model.eval()
            logging.info('Evaluating')
            val_NDCG, val_HR = val_eval(model)
            early_stopping(val_HR[10], model)
            if early_stopping.early_stop:
                break
            log_str = \
                f'Epoch {epoch:02d} | Valid | @3  | NDCG {val_NDCG[3]:.4f} | HR {val_HR[3]:.4f}\n' + \
                f'Epoch {epoch:02d} | Valid | @5  | NDCG {val_NDCG[5]:.4f} | HR {val_HR[5]:.4f}\n' + \
                f'Epoch {epoch:02d} | Valid | @10 | NDCG {val_NDCG[10]:.4f} | HR {val_HR[10]:.4f}\n' + \
                "=================================================="

            logging.info(log_str)
            model.train()

    return early_stopping