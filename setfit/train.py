import argparse

from flair.datasets import CONLL_03
from flair.models import TokenClassifier
from flair.embeddings import TransformerWordEmbeddings
from flair.trainers import ModelTrainer

from setfit.modeling import SetFitDecoder, SFTokenClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CONLL03")
    parser.add_argument("--label_type", type=str, default="ner")
    parser.add_argument("--transformer_model", type=str, default="bert-base-uncased")
    parser.add_argument("--tag_type", type=str, default="BIO")
    parser.add_argument("--save_path", type=str, default="resources/setfit/")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_size", type=int, default=4) # this always needs to be <= batch size

    args = parser.parse_args()

    # TODO: Implement datasets you want to use.
    # TODO: Downsample the dataset (train and validation split) with .downsample()
    if args.dataset == "CONLL03":
        dataset = CONLL_03()
    else:
        raise ValueError(f"Dataset {args.dataset} is unknown.")

    label_dictionary = dataset.make_label_dictionary(args.label_type)

    # TODO: Implement the embeddings you want to use.
    embeddings = TransformerWordEmbeddings(args.transformer_model)

    # TODO: 1st step: Implement SetFit Contrastive Pre-Training. Possible Implementations:
    # TODO: 1) You could try flair's decoder argument - Implement a pytorch module that does the first step of SetFit
    #  (contrasting entities with each other). See example in modeling.py.

    # TODO: 2) Write a new model that inherits from flair.models.Model.

    # TODO: You need to build the entity pairs and contrast them, save the encoder,
    #  but you do not need to perform final evaluation. This comes in the second step.

    # TODO: You can do so by deleting the test and dev split from Corpus instance

    setfit_decoder = SetFitDecoder(label_dictionary=label_dictionary)

    model = SFTokenClassifier(
        embeddings=embeddings,
        label_dictionary=label_dictionary,
        label_type=args.label_type,
        decoder=setfit_decoder,
        span_encoding="BIO"
    )

    trainer = ModelTrainer(model, dataset)

    trainer.fine_tune(
        args.save_path,
        learning_rate=args.learning_rate,
        mini_batch_size=args.batch_size,
        mini_batch_chunk_size=args.gradient_accumulation_size,
    )

    # TODO: load the pre-trained encoder - only the encoder!
    setfit_model = TokenClassifier.load(args.save_path)

    # TODO: Implement 2nd phase: fine-tuning. This is standard procedure, you can re-use existing scripts.
    model = TokenClassifier(
        setfit_model.embeddings, # only use the contrastive pretrained encoder
        label_dictionary,
        tag_type=args.tag_type,
        decoder=setfit_decoder
    )

    trainer = ModelTrainer(model, dataset)

    trainer.fine_tune(
        args.save_path,
        learning_rate=args.learning_rate,
        mini_batch_size=args.batch_size,
        mini_batch_chunk_size=args.gradient_accumulation_size,
    )

