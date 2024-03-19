from pathlib import Path

import flair
import wandb
from flair.models import TokenClassifier

# from flair.trainers import ModelTrainer
from setfit.trainers import ModelTrainer
from setfit.utils import init_wandb_logger, select_dataset, select_dataset_filtering, parse_training_arguments, \
    GLOBAL_PATHS
from setfit.wandb_logger import WandbLogger


def finetuning_training_loop():
    flair.set_seed(wandb.config.seed)

    dataset = select_dataset(wandb.config.dataset)

    select_dataset_filtering(dataset, wandb.config.filtering_method, k_shot_num)

    setfit_model_path = Path(GLOBAL_PATHS["contrastive_model_path"]) / GLOBAL_PATHS["contrastive_model_filename"]
    setfit_model = TokenClassifier.load(setfit_model_path)

    label_dictionary = dataset.make_label_dictionary("ner")

    model = TokenClassifier(
        embeddings=setfit_model.embeddings,  # only use the contrastive pretrained encoder
        label_dictionary=label_dictionary,
        label_type="ner",
        span_encoding=wandb.config.tag_type,
    )

    trainer = ModelTrainer(model, dataset)
    wandb_logger = WandbLogger(wandb=wandb)

    trainer.fine_tune(
        GLOBAL_PATHS["save_path"],
        learning_rate=wandb.config.learning_rate,
        max_epochs=wandb.config.max_epochs,
        mini_batch_size=wandb.config.batch_gradient_size[0],
        mini_batch_chunk_size=wandb.config.batch_gradient_size[1],
        plugins=[wandb_logger]
    )


if __name__ == "__main__":
    args = parse_training_arguments()
    init_wandb_logger(args, workflow="finetuning_only")
    flair.set_seed(wandb.config.seed)

    finetuning_training_loop()
