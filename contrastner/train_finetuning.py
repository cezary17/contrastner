from pathlib import Path
import random

import flair
import wandb
from flair.models import TokenClassifier
import numpy as np

from contrastner.dataset import KShotCounter
from contrastner.trainers import ModelTrainer
from contrastner.utils import init_wandb_logger, select_corpus, parse_training_arguments, GLOBAL_PATHS
from contrastner.wandb_logger import WandbLogger


def finetuning_training_loop():
    flair.set_seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
    random.seed(wandb.config.seed)

    corpus = select_corpus(wandb.config.dataset)

    k_shot_counter = KShotCounter(
        k=wandb.config.k_shot_num,
        mode=wandb.config.filtering_method,
        simple_cutoff=wandb.config.filtering_cutoff,
        remove_dev=True,
        shuffle=wandb.config.shuffle_dataset,
        shuffle_seed=wandb.config.seed
    )

    k_shot_counter(corpus)

    label_dictionary = corpus.make_label_dictionary(label_type="ner")
    setfit_model_path = Path(GLOBAL_PATHS["contrastive_model_path"]) / GLOBAL_PATHS["contrastive_model_filename"]
    setfit_model = TokenClassifier.load(setfit_model_path)

    model = TokenClassifier(
        embeddings=setfit_model.embeddings,  # only use the contrastive pretrained encoder
        label_dictionary=label_dictionary,
        label_type="ner",
        span_encoding=wandb.config.tag_type,
    )

    trainer = ModelTrainer(model, corpus)
    wandb_logger = WandbLogger(wandb=wandb)

    trainer.fine_tune(
        GLOBAL_PATHS["save_path"],
        learning_rate=wandb.config.learning_rate,
        max_epochs=wandb.config.max_epochs,
        mini_batch_size=wandb.config.batch_size,
        plugins=[wandb_logger]
    )


if __name__ == "__main__":
    args = parse_training_arguments()
    init_wandb_logger(args, workflow="finetuning_only")

    finetuning_training_loop()
