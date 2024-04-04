import logging

import wandb

from contrastner.train_baseline import baseline_train_loop
from contrastner.train_contrastive import contrastive_training_loop
from contrastner.train_finetuning import finetuning_training_loop

log = logging.getLogger("flair")

import argparse


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


if __name__ == "__main__":
    # train_full_sweep()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str)
    parser.add_argument("--count", type=int, default=10)

    args = parser.parse_args()
    wandb.agent(args.sweep_id, function=train_full_sweep, count=args.count)
