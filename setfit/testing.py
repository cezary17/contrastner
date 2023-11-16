from modeling import SetFitDecoder
from flair.datasets import CONLL_03

corpus = CONLL_03()
corpus = corpus.downsample(0.1)

one_example = corpus.train[0]
five_exampels = [corpus.train[i] for i in range(5)]

label_dictionary = corpus.make_label_dictionary(label_type="ner")

setfit = SetFitDecoder(label_dictionary=label_dictionary)

print(one_example)

ner_labels = setfit._extract_named_entities(one_example)

for label in ner_labels:
    print(label)
    print(label.data_point.unlabeled_identifier)
    print(label.value)