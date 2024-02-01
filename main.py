from os.path import join as path_join
import torch
import wandb
import logging
from pathlib import Path

from model import SASRec, Selector
from data import RecDataset, AugDataset
from utils import (data_partition, set_seed, load_eval, load_llm_generation, 
                   load_item_embedding, load_projection)
from evaluation import Evaluator

from config import get_args
from logger import create_logger
from trainer import train

args = get_args()

Path(path_join('experiments', args.dataset, args.exp_name)).mkdir(parents=True, exist_ok=True)
args.save_dir = path_join('experiments', args.dataset, args.exp_name)
Path(path_join(args.save_dir, 'log.log')).unlink(missing_ok=True)
logger = create_logger(path_join(args.save_dir, 'log.log'))

if args.wandb:
    wandb_name = args.exp_name
    wandb.init(project=args.wandb_project, name=wandb_name, group=args.group)
    wandb.config.update(args)

if __name__ == '__main__':
    # global dataset
    set_seed(args.seed)
    user_train, user_single, user_overlap, usernum, usernum_A, itemnum = data_partition(args.dataset, args.overlap_ratio)
    # user_valid, user_test, Anum, Bnum = load_eval(args.dataset)
    user_valid, user_test, Anum, Bnum = load_eval(args.dataset, max_len=args.maxlen)

    user_aug = load_llm_generation(args.dataset, user_single, usernum_A, args.backbone)
    A_item_embedding, B_item_embedding = load_item_embedding(args.dataset, args.device)
    projection = load_projection(args.dataset, args.device)
    aug_dataset = AugDataset(User=user_aug)

    pooling_model = Selector(
        hidden_units=768,
        model_hidden=args.hidden_units,
        k=args.k, 
        device=args.device, 
        itemnum=itemnum, 
        Anum=Anum,
        projection=projection, 
        item_embeddings=(A_item_embedding, B_item_embedding)).to(args.device)

    if args.pooler_init == 'xavier':
        torch.nn.init.xavier_normal_(pooling_model.pooler_A.weight.data)
        torch.nn.init.xavier_normal_(pooling_model.pooler_B.weight.data)
    elif args.pooler_init == 'zero':
        torch.nn.init.zeros_(pooling_model.pooler_A.weight.data)
        torch.nn.init.zeros_(pooling_model.pooler_B.weight.data)

    rec_dataset = RecDataset(
        User=user_train,
        usernum=usernum,
        itemnum=itemnum,
        Anum=Anum,
        maxlen=args.maxlen)
    meta_dataset = RecDataset(
        User=user_overlap,
        usernum=len(user_overlap),
        itemnum=itemnum,
        Anum=Anum,
        maxlen=args.maxlen)
    
    val_evaluator = Evaluator(user_valid, Anum, Bnum, args.maxlen, args.batch_size, args.device)
    test_evaluator = Evaluator(user_test, Anum, Bnum, args.maxlen, args.batch_size, args.device)

    model = SASRec(usernum, Anum, Bnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    
    early_stopping_A, early_stopping_B = train(
        model=model,
        pooling_model=pooling_model,
        rec_dataset=rec_dataset,
        aug_dataset=aug_dataset,
        meta_dataset=meta_dataset,
        evaluators=(val_evaluator, test_evaluator),
        args=args
    )

    model.eval()
    model.load_state_dict(early_stopping_A.best_model)
    test_NDCG_A, test_HR_A, _, _ = test_evaluator(model)
    model.load_state_dict(early_stopping_B.best_model)
    _, _, test_NDCG_B, test_HR_B = test_evaluator(model)

    final_key_map = {
        'A': {'NDCG': test_NDCG_A, 'HR': test_HR_A},
        'B': {'NDCG': test_NDCG_B, 'HR': test_HR_B},
    }
    if args.wandb:
        wandb_dict = {"epoch": 0, "best_val_A": early_stopping_A.best_score, "best_val_B": early_stopping_B.best_score}
        for domain in ['A', 'B']:
            for metric in ['NDCG', 'HR']:
                for k in [1, 3, 5, 10]:
                    wandb_dict.update({f"final_{domain}_{metric}@{k}": final_key_map[domain][metric][k]})
        wandb.log(wandb_dict)

    log_str = \
        f'Best | Test | A@1  | NDCG {test_NDCG_A[1]:.4f} | HR {test_HR_A[1]:.4f}\n' + \
        f'Best | Test | A@3  | NDCG {test_NDCG_A[3]:.4f} | HR {test_HR_A[3]:.4f}\n' + \
        f'Best | Test | A@5  | NDCG {test_NDCG_A[5]:.4f} | HR {test_HR_A[5]:.4f}\n' + \
        f'Best | Test | A@10 | NDCG {test_NDCG_A[10]:.4f} | HR {test_HR_A[10]:.4f}\n' + \
        f'Best | Test | B@1  | NDCG {test_NDCG_B[1]:.4f} | HR {test_HR_B[1]:.4f}\n' + \
        f'Best | Test | B@3  | NDCG {test_NDCG_B[3]:.4f} | HR {test_HR_B[3]:.4f}\n' + \
        f'Best | Test | B@5  | NDCG {test_NDCG_B[5]:.4f} | HR {test_HR_B[5]:.4f}\n' + \
        f'Best | Test | B@10 | NDCG {test_NDCG_B[10]:.4f} | HR {test_HR_B[10]:.4f}\n'
    logging.info(log_str)

