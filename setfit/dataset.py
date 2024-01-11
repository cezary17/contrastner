import typing
from collections import defaultdict

from flair.data import Corpus
from torch.utils.data.dataset import Subset


def filter_dataset(corpus: Corpus, fs_examples_num: int = 100) -> typing.NoReturn:
    """
    Filter a corpus to only contain sentences with at least 2 labeled entities.
    :param corpus: The corpus to filter.
    :return: None
    """
    indices = find_indices(corpus, fs_examples_num)
    corpus._train = Subset(corpus.train, indices)


def find_indices(corpus: Corpus, fs_examples_num: int = 100) -> typing.List[int]:
    """
        Find the first <count> indices in a corpus that have >= 2 labeled entities.
        :param corpus: The corpus to filter.
        :param count: The number of indices to find.
        :return: List of indices.
        """
    indices = []
    for sentence_index, sentence in enumerate(corpus.train):
        sentence_dict = sentence.to_dict(tag_type="ner")
        sentence_counter = defaultdict(int)

        for entity in sentence_dict["entities"]:
            max_confidence_label = max(entity["labels"], key=lambda x: x["confidence"])
            sentence_counter[max_confidence_label["value"]] += 1

        if any([count >= 2 for count in sentence_counter.values()]):
            indices.append(sentence_index)

        if len(indices) > fs_examples_num:
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
