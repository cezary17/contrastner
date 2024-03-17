import flair
import wandb
from flair.embeddings import TransformerWordEmbeddings

from setfit.dataset import remove_dev_and_train
from setfit.modeling import SFTokenClassifier
from setfit.trainers import ModelTrainer
from setfit.utils import select_dataset, select_dataset_filtering, parse_training_arguments, init_wandb_logger, \
    GLOBAL_PATHS
from setfit.wandb_logger import WandbLogger


# from flair.trainers import ModelTrainer


def setfit_training_loop():

    flair.set_seed(wandb.config.seed)

    dataset_name = wandb.config.dataset
    dataset = select_dataset(dataset_name)

    select_dataset_filtering(wandb.config.filtering_method, dataset)
    remove_dev_and_train(dataset)

    embeddings = TransformerWordEmbeddings(wandb.config.transformer_model)
    label_dictionary = dataset.make_label_dictionary("ner")

    # TODO: can be selected dynamically if not enough candidates found in the dataset
    contrast_filtering_method = wandb.config.filtering_method

    model = SFTokenClassifier(
        embeddings=embeddings,
        label_dictionary=label_dictionary,
        label_type="ner",
        span_encoding=wandb.config.tag_type,
        contrast_filtering_method=contrast_filtering_method
    )

    trainer = ModelTrainer(model, dataset)
    wandb_logger = WandbLogger(wandb=wandb)

    trainer.fine_tune(
        GLOBAL_PATHS["contrastive_model_path"],
        learning_rate=wandb.config.learning_rate,
        max_epochs=wandb.config.max_epochs,
        mini_batch_size=wandb.config.batch_gradient_size[0],
        mini_batch_chunk_size=wandb.config.batch_gradient_size[1],
        plugins=[wandb_logger]
    )


if __name__ == "__main__":
    args = parse_training_arguments()
    init_wandb_logger(args, workflow="contrastive_only")
    flair.set_seed(wandb.config.seed)

    setfit_training_loop()
