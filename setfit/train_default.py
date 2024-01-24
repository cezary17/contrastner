import argparse

import flair
from flair.datasets import CONLL_03
from flair.embeddings import TransformerWordEmbeddings
from flair.models import TokenClassifier
from flair.trainers import ModelTrainer

from setfit.dataset import filter_dataset


def default_train_loop():
    flair.set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CONLL03")
    parser.add_argument("--label_type", type=str, default="ner")
    parser.add_argument("--transformer_model", type=str, default="bert-base-uncased")
    parser.add_argument("--tag_type", type=str, default="BIO")
    parser.add_argument("--save_path", type=str, default="resources/setfit/default")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_size", type=int, default=4)  # this always needs to be <= batch size

    args = parser.parse_args()

    if args.dataset == "CONLL03":
        dataset = CONLL_03()
    else:
        raise ValueError(f"Dataset {args.dataset} is unknown.")

    label_dictionary = dataset.make_label_dictionary(label_type=args.label_type, add_unk=False)

    filter_dataset(dataset)

    embeddings = TransformerWordEmbeddings(args.transformer_model)

    model = TokenClassifier(
        embeddings=embeddings,  # only use the contrastive pretrained encoder
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
    )


if __name__ == "__main__":
    default_train_loop()
