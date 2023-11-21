import random

from modeling import SetFitDecoder
from flair.datasets import CONLL_03

corpus = CONLL_03()
corpus = corpus.downsample(0.1)

one_example = corpus.train[0]
five_exampels = [corpus.train[i] for i in range(5)]

label_dictionary = corpus.make_label_dictionary(label_type="ner")

setfit = SetFitDecoder(label_dictionary=label_dictionary)

print(one_example)

labels = one_example.to_dict(tag_type="ner")

possible_labels = ["LOC", "PER", "ORG", "MISC", "O"]
test_labels = [random.choice(possible_labels) for _ in range(10)]

ner_labels = setfit._make_entity_triplets(labels=test_labels)

for label, l in ner_labels.items():
    print(f"{label}: {l}")