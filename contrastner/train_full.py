import logging

import torch
import flair
import wandb

from contrastner.utils import parse_training_arguments, sweep_config, init_wandb_logger
from contrastner.train_baseline import baseline_train_loop
from contrastner.train_contrastive import contrastive_training_loop
from contrastner.train_finetuning import finetuning_training_loop

log = logging.getLogger("flair")


def train_full():
    init_wandb_logger(args, workflow="full")
    flair.device = torch.device(f"cuda:{args.device}")
    contrastive_training_loop()
    finetuning_training_loop()


if __name__ == "__main__":
    args = parse_training_arguments()
    if args.sweep:
        sweep_id = wandb.sweep(sweep=sweep_config(args), project="fsner")
    else:
        train_full()
