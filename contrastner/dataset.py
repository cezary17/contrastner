import logging
import typing
from collections import defaultdict

from flair.data import Corpus, Sentence
from torch.utils.data.dataset import Subset

log = logging.getLogger("flair")

import numpy as np


class KShotCounter:
    def __init__(
            self,
            k: int,
            mode: str,
            simple_cutoff: int = 1,
            remove_dev: bool = False,
            remove_test: bool = False,
            shuffle: bool = False,
            shuffle_seed: int = 0):
        """
        Counter for k-shot filtering.
        :param k: The number of instances for each label.
        :param mode: Filtering type. Possible values: "simple", "contrastive"
        :param simple_cutoff: Minimum number of instances for a label to be considered. Only used in "simple" mode.
        :param remove_dev: Remove the dev split from the corpus.
        :param remove_test: Remove the test split from the corpus.
        """
        self.k = k

        if mode not in ["simple", "contrastive"]:
            raise ValueError(f"Unknown mode {mode}")
        self.mode = mode

        self.simple_cutoff = simple_cutoff
        self.remove_dev = remove_dev
        self.remove_test = remove_test
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

        self.indices = []
        self.counted_labels = None
        self.labels = None
        self.iterating_order = None

    def __getitem__(self, key):
        return self.labels[key]

    def __setitem__(self, key, value):
        self.labels[key] = value

    def __iadd__(self, key, value):
        assert isinstance(value, int), "Can only add integers to KShotCounter"
        self[key] = self[key] + value

    def __isub__(self, key, other):
        assert isinstance(other, int), "Can only subtract integers from KShotCounter"
        self[key] = self[key] - other

    def values(self):
        return self.labels.values()

    @staticmethod
    def make_labels_dict(sentence: Sentence) -> typing.Dict[str, int]:

        sentence_dict = sentence.to_dict(tag_type="ner")
        labels_dict = defaultdict(int)

        for entity in sentence_dict["entities"]:
            max_confidence_label = max(entity["labels"], key=lambda x: x["confidence"])
            labels_dict[max_confidence_label["value"]] += 1

        return labels_dict

    def _check(self, labels: typing.Dict[str, int]) -> bool:
        if self.mode == "simple":
            return self._check_simple(labels)
        elif self.mode == "contrastive":
            return self._check_contrastable(labels)
        else:
            raise ValueError(f"ain't gon happen {self.mode}")

    @staticmethod
    def _check_contrastable(labels: typing.Dict[str, int]) -> bool:
        """
        Check if adding the given labels to the counter will exceed the k limit.
        Idea:
            Take sentence with 2 labels PER and some other label (ORG). -> 2 PER 1 ORG
            We can only contrast per with this org sentence -> 1 K example
            For now we only take sentences with exactly 2 labels to not cheat the results.
        :param labels: The labels to check.
        :return: True if the labels can be added without exceeding k, False otherwise.
        """
        # Find label with 2 instances
        condition_2_labels = any([count == 2 for count in labels.values()])
        # Assert no labels are represented more than 2 times (maybe not necessary)
        condition_no_exceeding_2 = all([count <= 2 for count in labels.values()])
        # Assert that we have exactly one label with 2 instances
        condition_excactly_2 = sum([count == 2 for count in labels.values()]) == 1
        # Assert that we have at least one label for contrasting
        condition_contrastable = sum([count >= 1 for count in labels.values()]) >= 2

        return condition_2_labels and condition_no_exceeding_2 and condition_excactly_2 and (
            condition_contrastable)

    def _check_simple(self, labels: typing.Dict[str, int]) -> bool:
        """
        Check if adding labels to counter will exceed the k limit.
        Here simple version -> just count labels and add those values to the counter.
        :param labels: The labels to check.
        :param cutoff: Minimum number of instances for a label to be considered.
        :return:
        """
        # Find label with at least 2 instances
        condition_2_labels = any([count >= self.simple_cutoff for count in labels.values()])

        return condition_2_labels

    def _add_to_counter(self, labels: typing.Dict[str, int]) -> bool:
        """
        Increment the counter with the given labels.
        :param labels:
        :return:
        """
        if self.mode == "simple":
            # do the loop twice to not add anything to the counter before we get through all labels
            return self._add_simple(labels)

        elif self.mode == "contrastive":
            return self._add_contrastive(labels)

    def _add_contrastive(self, labels):
        target_label = self._find_target_label(labels)
        if self[target_label] < self.k:
            self[target_label] += 1
            return True
        return False

    def _add_simple(self, labels):
        for label, count in labels.items():
            if self[label] + count > self.k:
                return False
        for label, count in labels.items():
            self[label] += count
        return True

    @staticmethod
    def _find_target_label(labels: typing.Dict[str, int]) -> str:
        """
        Find the label with 2 instances.
        :param labels: A dict of labels.
        :return: The label with 2 instances.
        """
        return max(labels, key=labels.get)

    def _add_sentence(self, labels: typing.Union[typing.Dict[str, int], Sentence]) -> bool:
        """
        Add a sentence to the counter.
        :param labels: The labels to add. Can be a dict of labels or a flair Sentence.
        :return: True if the sentence was added, False otherwise.
        """
        if isinstance(labels, Sentence):
            labels = self.make_labels_dict(labels)

        if not self._check(labels):
            return False

        return self._add_to_counter(labels)

    def _is_full(self) -> bool:
        overflow = any([count > self.k for count in self.values()])
        assert not overflow, "Counter overflow in is_full call."

        return all([count == self.k for count in self.values()])

    def _get_sum(self) -> int:
        return sum(self.values())

    def _find_indices(self, corpus: Corpus):
        log.info(f"Starting to find indices for k-shot filtering with k={self.k}")

        iteration = 0
        for sentence_index in self.iterating_order:
            sentence = corpus.train[sentence_index]
            labels_dict = self.make_labels_dict(sentence)

            if self._add_sentence(labels_dict):
                self.indices.append(sentence_index)

            if self._is_full():
                log.info(f"Counter is full, stopping early after {iteration} iterations")
                return self.indices

            iteration += 1

    def _prep_dataset(self, corpus: Corpus):
        self._find_indices(corpus)
        corpus._train = Subset(corpus.train, self.indices)
        if self.remove_dev:
            corpus._dev = None
        if self.remove_test:
            corpus._test = None

    def __call__(self, corpus: Corpus, *args, **kwargs):
        self.counted_labels = corpus.make_label_dictionary(label_type="ner").get_items()

        if self.shuffle:
            np.random.seed(self.shuffle_seed)
            self.iterating_order = np.random.permutation(len(corpus.train))
        else:
            self.iterating_order = np.arange(len(corpus.train))

        self.labels = defaultdict(int, {label: 0 for label in self.counted_labels})
        self._prep_dataset(corpus)

