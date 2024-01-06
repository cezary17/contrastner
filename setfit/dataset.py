from flair.data import Sentence
from torch.utils.data.dataset import ConcatDataset, Dataset, Subset
from collections import Counter, defaultdict
from flair.datasets.sequence_labeling import ColumnDataset

def filter_dataset(corpus):
    """
    Filter a corpus to only contain sentences that have at least one entity.
    :param corpus: The corpus to filter.
    :param split: The split to filter. Default: train.
    :return: The filtered corpus.
    """
    filtered = []
    # prev_sentence_count = corpus.train.datasets[0].total_sentence_count

    for sentence in corpus.train:
        sentence_dict = sentence.to_dict(tag_type="ner")
        sentence_counter = defaultdict(int)

        for entity in sentence_dict["entities"]:
            max_confidence_label = max(entity["labels"], key=lambda x: x["confidence"])
            sentence_counter[max_confidence_label["value"]] += 1

        for label, count in sentence_counter.items():
            if count >= 2:
                filtered.append(sentence)
                break

    return corpus
