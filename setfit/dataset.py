import typing
from collections import defaultdict, Counter

from flair.data import Corpus, Sentence
from torch.utils.data.dataset import Subset


class KShotCounter(Counter):
    def __init__(self, *args, k: int, labels: typing.List[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.counted_labels = labels
        self.update({label: 0 for label in labels})

    def __missing__(self, key):
        if key in self.counted_labels:
            return 0
        else:
            raise KeyError(key)  # Random label, should never happen

    # def _old_check_labels_below_k(self, labels: typing.Dict[str, int]) -> bool:
    #
    #     condition_not_exceeding_k = all([self[label] + count <= self.k for label, count in labels.items()])
    #     condition_geq_zero = all([count >= 0 for count in labels.values()])
    #     condition_greater_2_one = any([count >= 2 for count in labels.values()])
    #     condition_greater_1_two = sum([count >= 1 for count in labels.values()]) >= 2
    #
    #     return condition_not_exceeding_k and condition_geq_zero and condition_greater_2_one and condition_greater_1_two

    @staticmethod
    def check_contrastable(labels: typing.Dict[str, int]) -> bool:
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
        # Assert that we have at least one label for contrasting
        condition_contrastable = sum([count >= 1 for count in labels.values()]) >= 2

        return condition_2_labels and condition_no_exceeding_2 and condition_contrastable

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

        if not self.check_contrastable(labels):
            return False

        self.update(labels)
        return True

    # @staticmethod
    # def check_passable_sentence(labels: typing.Dict[str, int]) -> bool:
    #     """
    #     Check if a sentence is still passable, even if it exceeds the k number of labels.
    #     Passable sentence: At least 1 label with >= 2 counts, and at least 2 different labels.
    #     :param labels: The dict of labels to check.
    #     :return: True if the sentence is passable, False otherwise.
    #     """
    #
    #     condition_greater_2_one = any([count >= 2 for count in labels.values()])
    #     condition_greater_1_two = sum([count >= 1 for count in labels.values()]) >= 2
    #
    #     return condition_greater_2_one and condition_greater_1_two

    # def check_labels_increase_total(self, labels: typing.Dict[str, int]) -> bool:
    #
    #     current_sum = self.get_sum()
    #     removed_counter = self.state_after_removal(labels)
    #
    #     if not removed_counter._old_check_labels_below_k(labels):
    #         return False
    #
    #     removed_counter.try_add_labels(labels)
    #     return removed_counter.get_sum() > current_sum

    def is_filled(self) -> bool:
        return all([count == self.k for count in self.values()])

    def get_sum(self) -> int:
        return sum(self.values())

    # def state_after_removal(self, labels: typing.Dict[str, int]) -> "KShotCounter":
    #     new_counter = KShotCounter(k=self.k, labels=self.counted_labels)
    #     new_counter.update(self)
    #     new_counter.remove_labels(labels)
    #     return new_counter

    @staticmethod
    def make_labels_dict(sentence: Sentence) -> typing.Dict[str, int]:

        sentence_dict = sentence.to_dict(tag_type="ner")
        labels_dict = defaultdict(int)

        for entity in sentence_dict["entities"]:
            max_confidence_label = max(entity["labels"], key=lambda x: x["confidence"])
            labels_dict[max_confidence_label["value"]] += 1

        return labels_dict


def find_indices_kshot(corpus: Corpus, k: int) -> typing.List[int]:
    corpus_labels = corpus.make_label_dictionary(label_type="ner").get_items()

    indices = []
    counter = KShotCounter(k=k, labels=corpus_labels)

    for sentence_index, sentence in enumerate(corpus.train):
        labels_dict = KShotCounter.make_labels_dict(sentence)

        if counter.add_sentence(labels_dict):
            indices.append(sentence_index)

        # elif counter.check_passable_sentence(labels_dict):
        #     # Here the sentence is still a correct sentence but adding it would exceed the k limit
        #     # Idea -> check if we can remove a sentence and add this one instead increasing the total label count
        #     selected_sentences = [corpus.train[i] for i in indices]
        #
        #     for candidate_sentence in selected_sentences:
        #         labels_dict = KShotCounter.make_labels_dict(candidate_sentence)
        #
        #         if counter.check_labels_increase_total(labels_dict):
        #             indices.remove(sentence)
        #             counter.remove_labels(labels_dict)
        #
        #             if counter.try_add_labels(labels_dict):
        #                 indices.append(sentence_index)
        #             else:
        #                 raise ValueError("Error in replacement of sentence")
        #
        #             break

        if counter.is_filled():
            break

    assert counter.is_filled(), "Not enough sentences to satisfy k-shot criterion."
    return indices


def filter_dataset(corpus: Corpus, k: int = 5) -> typing.NoReturn:
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
