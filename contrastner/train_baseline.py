import random

import flair
import numpy as np
import wandb
from flair.embeddings import TransformerWordEmbeddings
from flair.models import TokenClassifier

from contrastner.dataset import KShotCounter
from contrastner.utils import select_corpus, GLOBAL_PATHS, init_wandb_logger, parse_training_arguments
from contrastner.wandb_logger import WandbLogger
from contrastner.trainers import ModelTrainer


def baseline_train_loop():
    SEED = random.randint(0, 1000)
    wandb.define_metric("seed")
    wandb.log({"seed": SEED})
    flair.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    corpus = select_corpus(wandb.config.dataset)
    label_dictionary = corpus.make_label_dictionary(label_type="ner")

    k_shot_counter = KShotCounter(
        k=wandb.config.k_shot_num,
        mode=wandb.config.filtering_method,
        simple_cutoff=wandb.config.filtering_cutoff,
        remove_dev=True,
        shuffle=wandb.config.shuffle_dataset,
        shuffle_seed=SEED
    )

    k_shot_counter(corpus)

    embeddings = TransformerWordEmbeddings(
        wandb.config.transformer_model,
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=False,
        use_crf=False,
        use_rnn=False,
        reproject_embeddings=False
    )

    model = TokenClassifier(
        embeddings=embeddings,
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
    init_wandb_logger(args, workflow="baseline")

    baseline_train_loop()
