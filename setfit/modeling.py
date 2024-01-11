import logging
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

# TODO: Turn this into a proper class this is getting too big imo
"""
Dict of
    {
        "LOC": [(batch_idx1, idx_in_sentence1, text1), Tensor1), ((batch_idx2, idx_in_sentence2, text2), Tensor2),
        ...
"""
LABEL_TENSOR_DICT = Dict[str, List[Tuple[Tuple[int, int, str], torch.Tensor]]]
TRIPLET = Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


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
        # Shape: 53 * 3 * 768
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
                "LOC": [((batch, sentence_idx, 0), Tensor1), ((batch, sentence_idx, 1), Tensor2)]
                "PER": [((batch, sentence_idx, 3), Tensor3), ((batch, sentence_idx, 4), Tensor4)]
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

    def _make_entity_triplets(self, labels_dict: LABEL_TENSOR_DICT) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This will generate the list of triplets for all labels in label_tensor_dict

        We iterate over the labels and make triplets for every entity we found
        :param: labels_dict: Output from _make_label_token_dict
        :return: List of triplets [(anchor, (positive, negative)), ...]
        """

        triplets = []
        for entity, tensor_list in labels_dict.items():
            # skip if we have no different positive for that label
            if len(tensor_list) < 2:
                continue

            for current_anchor_idx, current_anchor_tup in enumerate(tensor_list):
                anchor_label = entity
                anchor_tensor = current_anchor_tup[1]

                # any entity not sharing the same label
                negative_label = choice([label for label in self.bio_label_list if (
                        label != anchor_label and
                        len(labels_dict[label]) > 0)])

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
