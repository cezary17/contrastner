import flair
import wandb
from flair.embeddings import TransformerWordEmbeddings

from contrastner.dataset import KShotCounter
from contrastner.modeling import SFTokenClassifier
from contrastner.trainers import ModelTrainer
from contrastner.utils import select_corpus, parse_training_arguments, init_wandb_logger, GLOBAL_PATHS
from contrastner.wandb_logger import WandbLogger

from contrastner.analysis.sentence_logger import log_selected_sentences


def contrastive_training_loop():
    flair.set_seed(wandb.config.seed)

    corpus = select_corpus(wandb.config.dataset)

    k_shot_counter = KShotCounter(
        k=wandb.config.k_shot_num,
        mode=wandb.config.filtering_method,
        simple_cutoff=wandb.config.filtering_cutoff,
        remove_dev=True,
        remove_test=True,
        shuffle=wandb.config.shuffle_dataset,
        shuffle_seed=wandb.config.seed
    )

    k_shot_counter(corpus)

    log_selected_sentences(corpus)

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

    label_dictionary = corpus.make_label_dictionary(label_type="ner")
    contrast_filtering_method = wandb.config.contrast_filtering_method

    model = SFTokenClassifier(
        embeddings=embeddings,
        label_dictionary=label_dictionary,
        label_type="ner",
        span_encoding=wandb.config.tag_type,
        contrast_filtering_method=contrast_filtering_method,
        neg_o_prob=wandb.config.neg_o_prob
    )

    trainer = ModelTrainer(model, corpus)
    wandb_logger = WandbLogger(wandb=wandb)

    trainer.fine_tune(
        GLOBAL_PATHS["contrastive_model_path"],
        learning_rate=wandb.config.learning_rate,
        max_epochs=wandb.config.max_epochs,
        mini_batch_size=wandb.config.batch_size,
        plugins=[wandb_logger]
    )


if __name__ == "__main__":
    args = parse_training_arguments()
    init_wandb_logger(args, workflow="contrastive_only")
    flair.set_seed(wandb.config.seed)

    contrastive_training_loop()
