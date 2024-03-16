import argparse

import flair
import wandb
from flair.embeddings import TransformerWordEmbeddings
from flair.trainers import ModelTrainer

from setfit.dataset import remove_dev_and_train
from setfit.modeling import SFTokenClassifier
from setfit.utils import select_dataset, select_dataset_filtering, parse_training_arguments, init_wandb_logger
from setfit.wandb_logger import WandbLogger


def setfit_training_loop(args: argparse.Namespace):
    wandb_logger = WandbLogger(wandb=wandb)

    dataset = select_dataset(args)

    select_dataset_filtering(args, dataset)

    remove_dev_and_train(dataset)

    embeddings = TransformerWordEmbeddings(args.transformer_model)
    label_dictionary = dataset.make_label_dictionary(args.label_type)

    model = SFTokenClassifier(
        embeddings=embeddings,
        label_dictionary=label_dictionary,
        label_type=args.label_type,
        span_encoding=args.tag_type,
    )

    trainer = ModelTrainer(model, dataset)

    trainer.fine_tune(
        args.contrastive_model_path,
        learning_rate=args.learning_rate,
        mini_batch_size=args.batch_size,
        mini_batch_chunk_size=args.gradient_accumulation_size,
        plugins=[wandb_logger]
    )


if __name__ == "__main__":
    args = parse_training_arguments()
    init_wandb_logger(args, workflow="contrastive_only")
    flair.set_seed(args.seed)

    setfit_training_loop(args)
