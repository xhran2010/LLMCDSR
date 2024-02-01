from os.path import join as path_join
import torch
import wandb
import logging
from pathlib import Path

from model import SASRec
from data import RecDataset, SeqPTDataset, load_text_weight
from utils import set_seed
from evaluation import Evaluator

from config import get_args
from logger import create_logger
from trainer import train

args = get_args()
Path(path_join('experiments', args.dataset, args.exp_name)).mkdir(parents=True, exist_ok=True)
args.save_dir = path_join('experiments', args.dataset, args.exp_name)
Path(path_join(args.save_dir, 'log.log')).unlink(missing_ok=True)
logger = create_logger(path_join(args.save_dir, 'log.log'))


if __name__ == '__main__':
    # global dataset
    set_seed(args.seed)
    raw_dataset = SeqPTDataset(args, domain=args.domain)
    user_train, user_val = raw_dataset.user_train, raw_dataset.user_valid
    text_weight = load_text_weight(args.dataset, args.domain, args.device)
    
    rec_dataset = RecDataset(
        User=user_train,
        usernum=raw_dataset.usernum,
        itemnum=raw_dataset.itemnum,
        maxlen=args.maxlen)

    val_evaluator = Evaluator(user_val.values(), raw_dataset.itemnum, args.maxlen, args.device)

    model = SASRec(raw_dataset.usernum, raw_dataset.itemnum, text_weight, args).to(args.device)  # no ReLU activation in original SASRec implementation?
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    
    early_stopping = train(
        model=model,
        rec_dataset=rec_dataset,
        val_eval=val_evaluator,
        args=args
    )


