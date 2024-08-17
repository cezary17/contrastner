import argparse
import logging
import random
from enum import Enum

import flair
import wandb
import numpy as np
from flair.datasets import CONLL_03, WNUT_17, FEWNERD, ONTONOTES, NER_ENGLISH_MOVIE_SIMPLE, NER_ENGLISH_RESTAURANT

log = logging.getLogger("flair")


class AvailableDataset(Enum):
    CONLL03 = "CONLL03"
    WNUT17 = "WNUT17"
    FEWNERD = "FEWNERD"
    ONTONOTES = "ONTONOTES"
    NER_ENGLISH_MOVIE_SIMPLE = "NER_ENGLISH_MOVIE_SIMPLE"
    NER_ENGLISH_RESTAURANT = "NER_ENGLISH_RESTAURANT"


GLOBAL_PATHS = {
    "contrastive_model_path": "resources/contrastner/contrastive/",
    "contrastive_model_filename": "final-model.pt",
    "save_path": "resources/contrastner/finetune"
}


def parse_training_arguments():
    parser = argparse.ArgumentParser()
    # run settings
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--device", type=str, default="0")
    # experiment settings
    parser.add_argument("--dataset", type=str, default="CONLL03")
    parser.add_argument("--transformer_model", type=str, default="bert-base-uncased")
    parser.add_argument("--tag_type", type=str, default="BIO")
    parser.add_argument("--filtering_method", type=str, default="simple")  # contrastive, simple
    parser.add_argument("--k_shot_num", type=int, default=8)
    parser.add_argument("--filtering_cutoff", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contrast_filtering_method", type=str, default="no-o")
    parser.add_argument("--shuffle_dataset", action="store_true", default=True)
    parser.add_argument("--neg_o_prob", type=float, default=0.2)
    parser.add_argument("--loss_function", type=str, default="TripletMarginLoss")
    # hyperparameters
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)

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
            "filtering_cutoff": args.filtering_cutoff,
            "shuffle_dataset": args.shuffle_dataset,
            "neg_o_prob": args.neg_o_prob,
            "loss_function": args.loss_function,

            # hyperparameters
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            **kwargs
        }
    )


def sweep_config(args: argparse.Namespace):
    log.info(f"Creating sweep with args: {args}")
    return {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "test_score"},
        "parameters": {
            "dummy": {
                "values": np.arange(0, 10).tolist()
            },
            "run_type": {
                "value": "baseline"
            },
            "neg_o_prob": {
                "value": 0.0
            },
            "contrast_filtering_method": {
                "value": "no-o"
            },
            "loss_function": {
                "value": "tml"
            },
            "max_epochs": {
                "value": 50
            },
            "k_shot_num": {
                "values": [3, 5, 10, 20, 50]
            },
            "learning_rate": {
                "value": 1e-4
            },
            "batch_size": {
                "value": 32
            },
            "filtering_cutoff": {
                "value": args.filtering_cutoff
            },
            "shuffle_dataset": {
                "value": args.shuffle_dataset
            },
            "dataset": {
                "values": ["CONLL03", "NER_ENGLISH_RESTAURANT"]
            },
            "transformer_model": {
                "value": "bert-base-uncased"
            },
            "tag_type": {
                "value": args.tag_type
            },
            "filtering_method": {
                "value": args.filtering_method
            },
        }
    }


def select_corpus(dataset_name: str) -> flair.data.Corpus:
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
