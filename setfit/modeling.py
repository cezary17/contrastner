import logging
from enum import Enum
from random import choice
from typing import List, Tuple, Dict

import flair
import torch
from flair.data import Dictionary, DT, Token, Label
from flair.embeddings import TokenEmbeddings
from flair.models import TokenClassifier
from torch.nn import TripletMarginLoss

logger = logging.getLogger("flair")
logger.setLevel("DEBUG")

flair.device = torch.device("cuda:1")

"""
Dict of
    {
        "LOC": [(batch_idx1, idx_in_sentence1, text1), Tensor1), ((batch_idx2, idx_in_sentence2, text2), Tensor2),
        ...
"""
LABEL_TENSOR_DICT = Dict[str, List[Tuple[Tuple[int, int, str], torch.Tensor]]]
TRIPLET = Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class FilterMethod(Enum):
    LESS_2 = "less-2"
    NONE = "none"
    NO_O = "no-o"


class FilterNotImplementedError(Exception):
    pass


class SFTokenClassifier(TokenClassifier):

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

        self.loss_function = TripletMarginLoss()
        self.label_list = self.label_dictionary.get_items() + ["O"]
        self._internal_batch_counter = 0
        self.bio_label_list = self._make_bio_label_list()

    def forward_loss(self, sentences: List[DT]) -> Tuple[torch.Tensor, int]:
        # make a forward pass to produce embedded data points and labels
        sentences = [
            sentence for sentence in sentences if self._filter_data_point(sentence)]

        # get the data points for which to predict labels
        data_points = self._get_data_points_for_batch(sentences)
        if len(data_points) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # get their gold labels as a tensor
        label_tensor = self._prepare_label_tensor(data_points)
        if label_tensor.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # pass data points through network to get encoded data point tensor
        self.embeddings.embed(sentences)

        labels_dict = self._make_label_tensor_dict(sentences)

        # Apply filters
        labels_dict = self._filter_labels_dict(labels_dict, FilterMethod.NO_O)

        triplets = self._make_entity_triplets(labels_dict)

        self._internal_batch_counter += 1

        # calculate the loss
        return self.loss_function(*triplets), label_tensor.size(0)

    def _cut_label_prefix(self, label: Label) -> str:
        """
        Maybe possible that this function is redundant and flair provides the simple label somewhere

        :param label: Label with positional information (start, inside, ...)
        :return: Simple Label like "MISC" or "PER"
        """
        label_val = label.value

        # labels have to start with "S" or "I" followed by "-" followed by the label itself
        if label_val != "O":
            label_val = label_val.split("-")[-1]

        # sanity check for development
        assert label_val in self.bio_label_list

        return label_val

    def _make_label_tensor_dict(self, data_points: list[Token]) -> LABEL_TENSOR_DICT:
        """
        This will get the input tensor batch and output a dictionary of the labels alongside their index and the
        corresponding tensors

        :param data_points: list of data points (flair.data.Token) as seen in TokenClassifier forward loop
        :return: A dict mapping the label to tensors having that label alongside it's index eg:
            {
                "LOC": [((batch, token_idx, "sample_text"), Tensor1), ((batch, token_idx, "sample_text"), Tensor2)]
                "PER": [((batch, token_idx, "sample_text"), Tensor3), ((batch, token_idx, "sample_text"), Tensor4)]
                ...
            }
        """

        # Initialize dict of labels with labels found in label_dictionary
        labels_dict = {label: [] for label in self.bio_label_list}

        for dp_idx, dp_sentence in enumerate(data_points):
            for token_idx, token in enumerate(dp_sentence):
                token_label = token.get_label("ner")

                # Map to simple embeddings to get "LOC" from "S-LOC", ...
                token_cut_label = self._cut_label_prefix(token_label)

                labels_dict[token_cut_label].append((
                    (self._internal_batch_counter, token_idx, token.text),
                    token.embedding
                ))

        return labels_dict

    def _filter_labels_dict(self, labels_dict: LABEL_TENSOR_DICT, method: str | FilterMethod) -> LABEL_TENSOR_DICT:
        """
        This will filter the labels dict to only contain labels with more than one entry

        :param labels_dict: Output from _make_label_token_dictA
        :param method: Method to use for filtering
        :return: Filtered dict
        """

        if isinstance(method, str):
            method = FilterMethod(method)

        match method:
            case FilterMethod.NONE:
                return labels_dict
            case FilterMethod.NO_O:
                return self._no_o_filter(labels_dict)
            case FilterMethod.LESS_2:
                return self._less_2_filter(labels_dict)
            case _:
                raise FilterNotImplementedError(f"Method {method.value} is not known.")

    @staticmethod
    def _no_o_filter(labels_dict: LABEL_TENSOR_DICT) -> LABEL_TENSOR_DICT:
        """
        This will remove the "O" label from the labels_dict
        :param labels_dict: Labels dict from _make_label_token_dict
        :return:
        """
        labels_dict.pop("O")
        return labels_dict

    @staticmethod
    def _less_2_filter(labels_dict: LABEL_TENSOR_DICT) -> LABEL_TENSOR_DICT:
        """
        This will remove all labels with less than 2 entries
        :param labels_dict: Labels dict from _make_label_token_dict
        :return:
        """
        return {label: tensor_list for label, tensor_list in labels_dict.items() if len(tensor_list) > 2}

    def _make_entity_triplets(self, labels_dict: LABEL_TENSOR_DICT) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This will generate the list of triplets for all labels in label_tensor_dict

        We iterate over the labels and make triplets for every entity we found
        :param: labels_dict: Output from _make_label_token_dict
        :return: List of triplets [(anchor, (positive, negative)), ...]
        """

        triplets = []

        # here for the case we throw O-labels out or similar
        relevant_labels = [label for label in self.bio_label_list if label in labels_dict.keys()]

        for entity, tensor_list in labels_dict.items():
            # skip if we have no different positive for that label
            if len(tensor_list) < 2:
                continue

            for current_anchor_idx, current_anchor_tup in enumerate(tensor_list):
                anchor_label = entity
                anchor_tensor = current_anchor_tup[1]

                # any entity not sharing the same label

                possible_neg_labels = [label for label in relevant_labels if
                                       (label != anchor_label and len(labels_dict[label]) > 0)]

                # problem with 1 label having a lot of entries and the rest having 0 -> no negative
                # should be fixed by filtering the dataset now but can never be too safe
                assert len(possible_neg_labels) > 0

                negative_label = choice(possible_neg_labels)

                negative_tensor = choice(labels_dict[negative_label])[1]

                # any entity sharing label with anchor without same index

                available_positives = tensor_list[:current_anchor_idx] + tensor_list[current_anchor_idx + 1:]
                positive_tensor = choice(available_positives)[1]

                triplets.append((anchor_tensor, positive_tensor, negative_tensor))

        full_anchor_tensor = torch.stack([triplet[0] for triplet in triplets])
        full_positive_tensor = torch.stack([triplet[1] for triplet in triplets])
        full_negative_tensor = torch.stack([triplet[2] for triplet in triplets])

        return full_anchor_tensor, full_positive_tensor, full_negative_tensor

    def _make_bio_label_list(self):
        bio_labels = []
        for label in self.label_list:
            if label != "O":
                bio_labels.append(label[2:])
            else:
                bio_labels.append(label)

        return list(set(bio_labels))
