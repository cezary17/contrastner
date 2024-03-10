import argparse

import flair
import wandb
from flair.datasets import CONLL_03, WNUT_17, FEWNERD, ONTONOTES, NER_ENGLISH_MOVIE_SIMPLE, NER_ENGLISH_RESTAURANT
from flair.embeddings import TransformerWordEmbeddings
from flair.trainers import ModelTrainer

from setfit.dataset import filter_dataset, filter_dataset_old, remove_dev_and_train
from setfit.modeling import SFTokenClassifier
from setfit.wandb_logger import WandbLogger


# from flair.trainers.plugins.loggers.wandb import WandbLogger


def setfit_training_loop():
    flair.set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CONLL03")
    parser.add_argument("--label_type", type=str, default="ner")
    parser.add_argument("--transformer_model", type=str, default="bert-base-uncased")
    parser.add_argument("--tag_type", type=str, default="BIO")
    parser.add_argument("--save_path", type=str, default="resources/setfit/contrastive/")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_size", type=int, default=4)  # this always needs to be <= batch size
    parser.add_argument("--filtering_method", type=str, default="legacy")  # legacy, k-shot
    parser.add_argument("--k_shot_num", type=int, default=5)

    args = parser.parse_args()

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
            "k_shot_num": args.k_shot_num,
        }
    )

    wandb_logger = WandbLogger(wandb=wandb)

    # This looks amazing ikr
    if args.dataset == "CONLL03":
        dataset = CONLL_03()
    elif args.dataset == "WNUT17":
        dataset = WNUT_17()
    elif args.dataset == "FEWNERD":
        dataset = FEWNERD()
    elif args.dataset == "ONTONOTES":
        dataset = ONTONOTES()
    elif args.dataset == "NER_ENGLISH_MOVIE_SIMPLE":
        dataset = NER_ENGLISH_MOVIE_SIMPLE()
    elif args.dataset == "NER_ENGLISH_RESTAURANT":
        dataset = NER_ENGLISH_RESTAURANT()
    else:
        raise ValueError(f"Dataset {args.dataset} is unknown.")

    if args.filtering_method == "k-shot":
        filter_dataset(dataset)
    elif args.filtering_method == "legacy":
        filter_dataset_old(dataset)
    else:
        raise ValueError(f"Filtering method {args.filtering_method} is unknown.")

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
        args.save_path,
        learning_rate=args.learning_rate,
        mini_batch_size=args.batch_size,
        mini_batch_chunk_size=args.gradient_accumulation_size,
        plugins=[wandb_logger]
    )


if __name__ == "__main__":
    setfit_training_loop()
