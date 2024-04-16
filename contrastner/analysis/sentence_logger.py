import logging

import pandas as pd
import wandb
from flair.data import Corpus

from contrastner.dataset import KShotCounter

log = logging.getLogger("flair")


def log_selected_sentences(filtered_corpus: Corpus):
    log.info(f"Selected {len(filtered_corpus.train)} sentences for training")
    log.info("Selected sentences in train:")

    sentences_dict = {"sentence": [], "labels": []}
    for sentence in filtered_corpus.train:
        log.info(sentence.to_tagged_string(main_label="ner"))
        sentences_dict["sentence"].append(sentence.to_tagged_string(main_label="ner"))
        sentences_dict["labels"].append(KShotCounter.make_labels_dict(sentence))

    sentence_df = pd.DataFrame(columns=["sentence", "labels"])
    sentence_df["sentence"] = sentences_dict["sentence"]
    sentence_df["labels"] = sentences_dict["labels"]

    sentence_table = wandb.Table(dataframe=sentence_df)
    wandb.log({"sentences_used": sentence_table})
