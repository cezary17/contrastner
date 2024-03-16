import flair

from setfit.utils import parse_training_arguments, init_wandb_logger
from train_contrastive import setfit_training_loop
from train_finetuning import finetuning_training_loop

if __name__ == "__main__":
    args = parse_training_arguments()
    init_wandb_logger(args, workflow="full")
    flair.set_seed(args.seed)

    setfit_training_loop(args)
    finetuning_training_loop(args)
