import typing
from typing import List, Tuple

import flair
import numpy as np
import torch
from flair.data import Sentence, Dictionary, DT
from flair.embeddings import TokenEmbeddings
from flair.models import TokenClassifier


class SFTokenClassifier(TokenClassifier):
    """
    Idea here is to change one thing in the forward model inherited from DefaultClassifier -> pass the labels tensor to
    the decoder
    """

    def __init__(
            self,
            embeddings: TokenEmbeddings,
            label_dictionary: Dictionary,
            label_type: str,
            span_encoding: str = "BIOES",
            **classifierargs, ) -> None:

        super().__init__(
            embeddings=embeddings,
            label_dictionary=label_dictionary,
            label_type=label_type,
            span_encoding=span_encoding,
            **classifierargs
        )

    def forward_loss(self, sentences: List[DT]) -> Tuple[torch.Tensor, int]:
        # make a forward pass to produce embedded data points and labels
        sentences = [sentence for sentence in sentences if self._filter_data_point(sentence)]

        # get the data points for which to predict labels
        data_points = self._get_data_points_for_batch(sentences)
        if len(data_points) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # get their gold labels as a tensor
        label_tensor = self._prepare_label_tensor(data_points)
        if label_tensor.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # pass data points through network to get encoded data point tensor
        data_point_tensor = self._encode_data_points(sentences, data_points)

        # HUGE CHANGES HERE!
        scores = self.decoder(data_point_tensor, label_tensor)

        # an optional masking step (no masking in most cases)
        scores = self._mask_scores(scores, data_points)

        # calculate the loss
        return self._calculate_loss(scores, label_tensor)


class SetFitDecoder(torch.nn.Module):

    def __init__(self, fs_examples: list[Sentence] = None, label_dictionary: Dictionary = None, *args, **kwargs):
        """
        who even reads docstrings?

        :param fs_examples: a list of Sentences -> Few Shot Examples
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.fs_examples = fs_examples
        self.label_dictionary = label_dictionary
        self.linear = torch.nn.Linear(768, 768)  # Just to throw you off that i know what i am doing

    def forward(self, inputs: torch.Tensor, label_tensor: typing.Any):
        """
        Should we really use this as a decoder module? If I understand what the decoder does in flair it is the last
        layer before loss is calculated. I thought the setfit contrastive loss function should be the last thing before
        the output to use the "shifted" vectors as embeddings for the "normal" fine-tuning.


        QUESTION FOR MEETING TOMORROW

        how to move vectors together/apart based on the groups

        :param label_tensor: tags of the input
        :param inputs: tensor of some shape
        :return:
        """

        labels_in_order = ["bruh"] * len(label_tensor)

        # problem here -> label index is not (always) the same as the index from label_dictionary
        for i, label_idx in enumerate(label_tensor):
            temp = self.label_dictionary.get_item_for_index(label_idx)
            labels_in_order[i] = temp

        return inputs

    def _extract_named_entities(self, sentence: Sentence):
        """
        This will get the Non-O entities from a sentence and return them as a list of tuples (entity, label, index)

        :param sentence:
        :return:
        """
        ner_labels = sentence.get_labels("ner")
        return ner_labels

    def _make_entity_triplets(self, sentence_list: list[Sentence]):
        pass
