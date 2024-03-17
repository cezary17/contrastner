import argparse
from pathlib import Path

import flair
import wandb
from flair.models import TokenClassifier

# from flair.trainers import ModelTrainer
from setfit.trainers import ModelTrainer
from setfit.utils import init_wandb_logger, select_dataset, select_dataset_filtering, parse_training_arguments
from setfit.wandb_logger import WandbLogger


def finetuning_training_loop(args: argparse.Namespace):
    wandb_logger = WandbLogger(wandb=wandb)

    dataset = select_dataset(args)
    select_dataset_filtering(args, dataset)

    setfit_model_path = Path(args.contrastive_model_path) / args.contrastive_model_filename
    setfit_model = TokenClassifier.load(setfit_model_path)

    label_dictionary = dataset.make_label_dictionary(args.label_type)

    model = TokenClassifier(
        embeddings=setfit_model.embeddings,  # only use the contrastive pretrained encoder
        label_dictionary=label_dictionary,
        label_type=args.label_type,
        span_encoding=args.tag_type,
    )

    trainer = ModelTrainer(model, dataset)

    trainer.fine_tune(
        args.save_path,
        learning_rate=args.learning_rate,
        mini_batch_size=args.batch_size,
        mini_batch_chunk_size=args.gradient_accumulation_size,
        plugins=[wandb_logger]
    )


if __name__ == "__main__":
    args = parse_training_arguments()
    init_wandb_logger(args, workflow="finetuning_only")
    flair.set_seed(args.seed)

    finetuning_training_loop(args)
