import wandb

from setfit.utils import parse_training_arguments, sweep_config, init_wandb_logger
from train_contrastive import contrastive_training_loop
from train_finetuning import finetuning_training_loop


def train_full():
    init_wandb_logger(args, workflow="full")
    contrastive_training_loop()
    finetuning_training_loop()


def train_full_sweep():
    with wandb.init() as run:
        contrastive_training_loop()
        finetuning_training_loop()


if __name__ == "__main__":
    args = parse_training_arguments()
    if args.sweep:
        sweep_id = wandb.sweep(sweep=sweep_config(args), project="fsner")
        wandb.agent(sweep_id, function=train_full_sweep, count=10)
    else:
        train_full()
