import flair
import wandb
from flair.embeddings import TransformerWordEmbeddings
from flair.models import TokenClassifier
from flair.trainers import ModelTrainer

from contrastner.utils import select_dataset, select_dataset_filtering, filter_dataset, GLOBAL_PATHS
from contrastner.wandb_logger import WandbLogger


def baseline_train_loop():
    flair.set_seed(wandb.config.seed)

    dataset = select_dataset(wandb.config.dataset)
    select_dataset_filtering(dataset, wandb.config.filtering_method, wandb.config.k_shot_num)

    label_dictionary = dataset.make_label_dictionary(label_type="ner")

    embeddings = TransformerWordEmbeddings(wandb.config.transformer_model)

    model = TokenClassifier(
        embeddings=embeddings,  # only use the contrastive pretrained encoder
        label_dictionary=label_dictionary,
        label_type="ner",
        span_encoding=wandb.config.tag_type,
    )

    trainer = ModelTrainer(model, dataset)
    wandb_logger = WandbLogger(wandb=wandb)

    trainer.fine_tune(
        GLOBAL_PATHS["save_path"],
        max_epochs=wandb.config.max_epochs,
        mini_batch_size=wandb.config.batch_size,
        plugins=[wandb_logger]
    )


if __name__ == "__main__":
    baseline_train_loop()
