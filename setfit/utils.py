import argparse
import logging
from enum import Enum

import flair
import wandb
from flair.datasets import CONLL_03, WNUT_17, FEWNERD, ONTONOTES, NER_ENGLISH_MOVIE_SIMPLE, NER_ENGLISH_RESTAURANT

from setfit.dataset import filter_dataset, filter_dataset_old
from setfit.wandb_logger import WandbLogger

log = logging.getLogger("flair")


class AvailableDataset(Enum):
    CONLL03 = "CONLL03"
    WNUT17 = "WNUT17"
    FEWNERD = "FEWNERD"
    ONTONOTES = "ONTONOTES"
    NER_ENGLISH_MOVIE_SIMPLE = "NER_ENGLISH_MOVIE_SIMPLE"
    NER_ENGLISH_RESTAURANT = "NER_ENGLISH_RESTAURANT"


GLOBAL_PATHS = {
    "contrastive_model_path": "resources/setfit/contrastive/",
    "contrastive_model_filename": "final-model.pt",
    "save_path": "resources/setfit/finetune"
}


def parse_training_arguments():
    parser = argparse.ArgumentParser()
    # run settings
    parser.add_argument("--sweep", action="store_true")
    # experiment settings
    parser.add_argument("--dataset", type=str, default="CONLL03")
    parser.add_argument("--transformer_model", type=str, default="bert-base-uncased")
    parser.add_argument("--tag_type", type=str, default="BIO")
    parser.add_argument("--filtering_method", type=str, default="k-shot")  # legacy, k-shot
    parser.add_argument("--k_shot_num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contrast_filtering_method", type=str, default="no-o")
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
            # experiment settings
            "dataset": args.dataset,
            "transformer_model": args.transformer_model,
            "tag_type": args.tag_type,
            "filtering_method": args.filtering_method,
            "k_shot_num": args.k_shot_num,
            "seed": args.seed,
            "contrast_filtering_method": args.contrast_filtering_method,

            # hyperparameters
            "learning_rate": args.learning_rate,
            "batch_gradient_size": (args.batch_size, args.gradient_accumulation_size),
            "max_epochs": args.max_epochs,
            **kwargs
        }
    )

    wandb_logger = WandbLogger(wandb=wandb)
    return wandb_logger


def sweep_config(args: argparse.Namespace):
    log.info(f"Creating sweep with args: {args}")
    return {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "dev/macro avg/f1-score"},
        "parameters": {
            "max_epochs": {
                "values": [1]
            },
            "learning_rate": {
                "values": [1e-4]
            },
            "batch_gradient_size": {
                "values": [(4, 4), (8, 8)]
            },
            "dataset": {
                "value": args.dataset
            },
            "transformer_model": {
                "value": args.transformer_model
            },
            "tag_type": {
                "value": args.tag_type
            },
            "filtering_method": {
                "value": args.filtering_method
            },
            "k_shot_num": {
                "value": args.k_shot_num
            },
            "contrast_filtering_method": {
                "value": args.contrast_filtering_method
            },
            "seed": {
                "value": args.seed
            }

        }
    }


def select_dataset(dataset_name: str):
    logging.info(f"Selecting dataset {dataset_name}")
    match AvailableDataset[dataset_name.upper()]:
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
            raise ValueError(f"Dataset {dataset_name} is unknown.")

    return dataset


def select_dataset_filtering(dataset: flair.data.Corpus, filter_type: str, k: int):
    logging.info(f"Selecting dataset filtering method with method {filter_type}")
    if filter_type == "k-shot":
        filter_dataset(dataset, k)
    elif filter_type == "legacy":
        filter_dataset_old(dataset)
    else:
        raise ValueError(f"Filtering method {filter_type} is unknown.")
