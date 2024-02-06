from collections import defaultdict
import typing
import json
from flair.data import Corpus
from flair.datasets import CONLL_03
import torch

dataset = CONLL_03()

full_dataset = torch.utils.data.dataset.ConcatDataset([dataset.train, dataset.dev, dataset.test])

label_counts = defaultdict(int)

for sentence in full_dataset:
    sentence_dict = sentence.to_dict(tag_type="ner")

    token_count = len(sentence_dict["tokens"])
    entity_count = len(sentence_dict["entities"])

    label_counts["O"] += token_count - entity_count

    for entity in sentence_dict["entities"]:
        max_confidence_label = max(entity["labels"], key=lambda x: x["confidence"])
        label_counts[max_confidence_label["value"]] += 1

print(dict(label_counts))

with open("label_counts.json", "w") as f:
    json.dump(dict(label_counts), f)