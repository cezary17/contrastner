import torch.cuda
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline
import evaluate
import numpy as np
import argparse

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
wnut = load_dataset("wnut_17")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval")
label_list = wnut["train"].features[f"ner_tags"].feature.names

id2label = {
    0: "O",
    1: "B-corporation",
    2: "I-corporation",
    3: "B-creative-work",
    4: "I-creative-work",
    5: "B-group",
    6: "I-group",
    7: "B-location",
    8: "I-location",
    9: "B-person",
    10: "I-person",
    11: "B-product",
    12: "I-product",
}
label2id = {
    "O": 0,
    "B-corporation": 1,
    "I-corporation": 2,
    "B-creative-work": 3,
    "I-creative-work": 4,
    "B-group": 5,
    "I-group": 6,
    "B-location": 7,
    "I-location": 8,
    "B-person": 9,
    "I-person": 10,
    "B-product": 11,
    "I-product": 12,
}

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def train_main():
    print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
    tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
    # example = wnut["train"][0]
    # labels = [label_list[i] for i in example["ner_tags"]]
    # [print(f"token: {token}  label: {label}") for token, label in zip(example["tokens"], labels)]

    model = AutoModelForTokenClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="model_output",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_wnut["train"],
        eval_dataset=tokenized_wnut["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def inference_main():
    text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
    classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
    model_output = classifier(text)
    for token in model_output:
        print(token, "\n")


if __name__ == "__main__":

    argparse = argparse.ArgumentParser()
    argparse.add_argument("-i", "--inference", action="store_true")

    args = argparse.parse_args()

    if args.inference:
        inference_main()
    else:
        train_main()