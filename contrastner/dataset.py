import logging
import typing
from collections import defaultdict, Counter

from flair.data import Corpus, Sentence
from torch.utils.data.dataset import Subset

log = logging.getLogger("flair")


class KShotCounter(Counter):
    def __init__(self, *args, k: int, labels: typing.List[str], mode: str, **kwargs):
        """
        Counter for k-shot filtering.
        :param args:
        :param k:
        :param labels:
        :param mode: Possible values: "simple", "contrastive"
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.k = k
        self.counted_labels = labels
        self.update({label: 0 for label in labels})

        if mode not in ["simple", "contrastive"]:
            raise ValueError(f"Unknown mode {mode}")
        self.mode = mode

        # if not enough workable labels exist we can try contrasting with O-Tokens
        self.allow_o_contrast = False

    def __missing__(self, key):
        if key in self.counted_labels:
            return 0
        else:
            raise KeyError(key)  # Random label, should never happen

    @staticmethod
    def make_labels_dict(sentence: Sentence) -> typing.Dict[str, int]:

        sentence_dict = sentence.to_dict(tag_type="ner")
        labels_dict = defaultdict(int)

        for entity in sentence_dict["entities"]:
            max_confidence_label = max(entity["labels"], key=lambda x: x["confidence"])
            labels_dict[max_confidence_label["value"]] += 1

        return labels_dict

    def check(self, labels: typing.Dict[str, int]) -> bool:
        if self.mode == "simple":
            return self.check_simple(labels)
        elif self.mode == "contrastive":
            return self.check_contrastable(labels)
        else:
            raise ValueError(f"ain't gon happen {self.mode}")

    def check_contrastable(self, labels: typing.Dict[str, int]) -> bool:
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
                condition_contrastable or not self.allow_o_contrast)

    @staticmethod
    def check_simple(labels: typing.Dict[str, int]) -> bool:
        """
        Check if adding labels to counter will exceed the k limit.
        Here simple version -> just count labels and add those values to the counter.
        :param labels:
        :return:
        """
        # Find label with at least 2 instances
        condition_2_labels = any([count >= 2 for count in labels.values()])

        return condition_2_labels

    def _add_to_counter(self, labels: typing.Dict[str, int]) -> bool:
        """
        Increment the counter with the given labels.
        :param labels:
        :return:
        """
        if self.mode == "simple":
            # do the loop twice to not add anything to the counter before we get through all labels
            for label, count in labels.items():
                if self[label] + count > self.k:
                    return False
            for label, count in labels.items():
                self[label] += count
            return True

        elif self.mode == "contrastive":
            target_label = self.find_target_label(labels)
            if self[target_label] < self.k:
                self[target_label] += 1
                return True
            return False

    @staticmethod
    def find_target_label(labels: typing.Dict[str, int]) -> str:
        """
        Find the label with 2 instances.
        :param labels: A dict of labels.
        :return: The label with 2 instances.
        """
        return max(labels, key=labels.get)

    def add_sentence(self, labels: typing.Union[typing.Dict[str, int], Sentence]) -> bool:
        """
        Add a sentence to the counter.
        :param labels: The labels to add. Can be a dict of labels or a flair Sentence.
        :return: True if the sentence was added, False otherwise.
        """
        if isinstance(labels, Sentence):
            labels = self.make_labels_dict(labels)

        if not all([label in self.counted_labels for label in labels.keys()]):
            unexpected_labels = [label for label in labels.keys() if label not in self.counted_labels]
            raise KeyError(f"Unexpected labels found in input: {unexpected_labels}")

        if not self.check(labels):
            return False

        return self._add_to_counter(labels)

    def is_full(self) -> bool:
        overflow = any([count > self.k for count in self.values()])
        assert not overflow, "Counter overflow in is_full call."

        return all([count == self.k for count in self.values()])

    def get_sum(self) -> int:
        return sum(self.values())


def find_indices_kshot(corpus: Corpus, k: int, allow_o_contrast: bool = False) -> typing.List[int]:
    log.info(f"Starting to find indices for k-shot filtering with k={k}")
    corpus_labels = corpus.make_label_dictionary(label_type="ner").get_items()
    indices = []
    counter = KShotCounter(k=k, labels=corpus_labels, mode="contrastive")

    for sentence_index, sentence in enumerate(corpus.train):
        labels_dict = KShotCounter.make_labels_dict(sentence)

        if counter.add_sentence(labels_dict):
            indices.append(sentence_index)

        if counter.is_full():
            log.info(f"Counter is full, stopping early after {sentence_index} iterations")
            return indices

    if not counter.is_full() and allow_o_contrast:
        log.info("Not enough sentences to satisfy k-shot criterion. State: {counter}")
        log.info("Trying to find sentences with O contrast.")
        counter.allow_o_contrast = True

        for sentence_index, sentence in enumerate(corpus.train):
            labels_dict = KShotCounter.make_labels_dict(sentence)

            if counter.add_sentence(labels_dict):
                indices.append(sentence_index)

            if counter.is_full():
                log.info(
                    f"Counter is full after allowing O contrasting. Stopping early after {sentence_index} iterations.")
                return indices

    raise ValueError("Not enough sentences to satisfy k-shot criterion.")


def find_indices_old(corpus: Corpus, fs_sentences_num: int) -> typing.List[int]:
    """
        Will keep it here until kshot version works as intended.
        Find the first <count> indices in a corpus that have >= 2 labeled entities.
        :param corpus: The corpus to filter.
        :param fs_sentences_num: Number of sentences to find.
        :return: List of indices.
        """
    indices = []
    for sentence_index, sentence in enumerate(corpus.train):
        sentence_dict = sentence.to_dict(tag_type="ner")
        sentence_counter = defaultdict(int)

        for entity in sentence_dict["entities"]:
            max_confidence_label = max(entity["labels"], key=lambda x: x["confidence"])
            sentence_counter[max_confidence_label["value"]] += 1

        greater_2_labels = [count >= 2 for count in sentence_counter.values()]
        greater_1_labels = [count >= 1 for count in sentence_counter.values()]

        # make sure we have something to get a negative sample from
        if any(greater_2_labels) and sum(greater_1_labels) >= 2:
            indices.append(sentence_index)

        if len(indices) > fs_sentences_num:
            break

    return indices


def remove_dev_and_train(corpus: Corpus) -> typing.NoReturn:
    """
    Remove the dev and test split from a corpus.
    :param corpus: The corpus to remove the dev and test split from.
    :return: None
    """
    corpus._dev = None
    corpus._test = None


def filter_dataset(corpus: Corpus, k: int) -> typing.NoReturn:
    """
    Filter a corpus to only contain sentences with at least 2 labeled entities.
    :param corpus: The corpus to filter.
    :param k: Number labels for low-resource task.
    :return: None
    """

    indices = find_indices_kshot(corpus, k)
    corpus._train = Subset(corpus.train, indices)


def filter_dataset_old(corpus: Corpus) -> typing.NoReturn:
    """
    Filter a corpus to only contain sentences with at least 2 labeled entities.
    :param corpus: The corpus to filter.
    :return: None
    """
    indices = find_indices_old(corpus, 20)
    corpus._train = Subset(corpus.train, indices)
