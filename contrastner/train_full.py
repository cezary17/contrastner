import logging

import wandb

from contrastner.utils import parse_training_arguments, sweep_config, init_wandb_logger
from contrastner.train_baseline import baseline_train_loop
from contrastner.train_contrastive import contrastive_training_loop
from contrastner.train_finetuning import finetuning_training_loop

log = logging.getLogger("flair")


def train_full():
    init_wandb_logger(args, workflow="full")
    contrastive_training_loop()
    finetuning_training_loop()


def train_full_sweep():
    with wandb.init() as run:
        if wandb.config.run_type == "baseline":
            log.info("Running baseline training")
            baseline_train_loop()
        elif wandb.config.run_type == "contrastive":
            log.info("Running contrastive training")
            contrastive_training_loop()
            finetuning_training_loop()
        else:
            raise ValueError(f"Unknown run type: {wandb.config.run_type}")


def train_full_sweep_dbg():
    with wandb.init() as run:
        contrastive_training_loop()
        finetuning_training_loop()


if __name__ == "__main__":
    args = parse_training_arguments()
    if args.sweep:
        sweep_id = wandb.sweep(sweep=sweep_config(args), project="fsner")
        # wandb.agent(sweep_id, function=train_full_sweep, count=10)
    else:
        train_full()
