import argparse
from enum import Enum

import wandb
from flair.datasets import CONLL_03, WNUT_17, FEWNERD, ONTONOTES, NER_ENGLISH_MOVIE_SIMPLE, NER_ENGLISH_RESTAURANT

from setfit.dataset import filter_dataset, filter_dataset_old
from setfit.wandb_logger import WandbLogger


class AvailableDataset(Enum):
    CONLL03 = "CONLL03"
    WNUT17 = "WNUT17"
    FEWNERD = "FEWNERD"
    ONTONOTES = "ONTONOTES"
    NER_ENGLISH_MOVIE_SIMPLE = "NER_ENGLISH_MOVIE_SIMPLE"
    NER_ENGLISH_RESTAURANT = "NER_ENGLISH_RESTAURANT"


def parse_training_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CONLL03")
    parser.add_argument("--label_type", type=str, default="ner")
    parser.add_argument("--transformer_model", type=str, default="bert-base-uncased")
    parser.add_argument("--contrastive_model_path", type=str, default="resources/setfit/contrastive/")  # fine-tuning only
    parser.add_argument("--contrastive_model_filename", type=str, default="final-model.pt")  # fine-tuning only
    parser.add_argument("--save_path", type=str, default="resources/setfit/finetune")  # output of fine-tuning
    # experiment settings
    parser.add_argument("--tag_type", type=str, default="BIO")
    parser.add_argument("--k_shot_num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--filtering_method", type=str, default="k-shot")  # legacy, k-shot
    # hyperparameters
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_size", type=int, default=4)  # this always needs to be <= batch size

    return parser.parse_args()


def init_wandb_logger(args: argparse.Namespace, **kwargs):
    wandb.init(
        project="fsner",
        config={
            "dataset": args.dataset,
            "label_type": args.label_type,
            "transformer_model": args.transformer_model,
            "tag_type": args.tag_type,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_size": args.gradient_accumulation_size,
            "filtering_method": args.filtering_method,
            "max_epochs": args.max_epochs,
            "k_shot_num": args.k_shot_num,
            "seed": args.seed,
            **kwargs
        }
    )

    wandb_logger = WandbLogger(wandb=wandb)
    return wandb_logger


def select_dataset(args: argparse.Namespace):
    match AvailableDataset[args.dataset.upper()]:
        case AvailableDataset.CONLL03:
            dataset = CONLL_03()
        case AvailableDataset.WNUT17:
            dataset = WNUT_17()
        case AvailableDataset.FEWNERD:
            dataset = FEWNERD()
        case AvailableDataset.ONTONOTES:
            dataset = ONTONOTES()
        case AvailableDataset.NER_ENGLISH_MOVIE_SIMPLE:
            dataset = NER_ENGLISH_MOVIE_SIMPLE()
        case AvailableDataset.NER_ENGLISH_RESTAURANT:
            dataset = NER_ENGLISH_RESTAURANT()
        case _:
            raise ValueError(f"Dataset {args.dataset} is unknown.")

    return dataset


def select_dataset_filtering(args: argparse.Namespace, dataset):
    if args.filtering_method == "k-shot":
        filter_dataset(dataset)
    elif args.filtering_method == "legacy":
        filter_dataset_old(dataset)
    else:
        raise ValueError(f"Filtering method {args.filtering_method} is unknown.")

def wandb_sweep_init():
    pass