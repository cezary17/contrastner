import logging
import random
from collections import defaultdict
from enum import Enum
from random import choice
from typing import List, Tuple, Dict

import flair
import torch
import wandb
from flair.data import Dictionary, DT, Token, Label
from flair.embeddings import TokenEmbeddings
from flair.models import TokenClassifier
from torch.nn import TripletMarginLoss, CosineEmbeddingLoss

log = logging.getLogger("flair")
log.setLevel("DEBUG")

flair.device = torch.device("cuda:1")

"""
Dict of
    {
        "LOC": [(batch_idx1, idx_in_sentence1, text1), Tensor1), ((batch_idx2, idx_in_sentence2, text2), Tensor2),
        ...
"""
LABEL_TENSOR_DICT = Dict[str, List[Tuple[Tuple[int, int, str], torch.Tensor]]]
TRIPLET = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class FilterMethod(Enum):
    LESS_2 = "less-2"
    NONE = "none"
    NO_O = "no-o"
    O_ONLY_NEG = "o-only-neg"


class FilterNotImplementedError(Exception):
    pass


class SFTokenClassifier(TokenClassifier):

    def __init__(
            self,
            embeddings: TokenEmbeddings,
            label_dictionary: Dictionary,
            label_type: str,
            span_encoding: str = "BIOES",
            contrast_filtering_method: str = "no-o",
            neg_o_prob: float = 0.2,
            loss_function: str = "TripletMarginLoss",
            **classifierargs, ) -> None:

        super().__init__(
            embeddings=embeddings,
            label_dictionary=label_dictionary,
            label_type=label_type,
            span_encoding=span_encoding,
            **classifierargs
        )

        loss_function = loss_function.strip().lower()
        if loss_function == "tripletmarginloss" or loss_function == "tml":
            self.loss_function = TripletMarginLoss()
            self.loss_used = "tml"
        elif loss_function == "cosineembeddingloss" or loss_function == "cel":
            self.loss_function = CosineEmbeddingLoss()
            self.loss_used = "cel"
        else:
            raise NotImplementedError(f"Loss function {loss_function} is not implemented")

        self.label_list = self.label_dictionary.get_items() + ["O"]
        self._internal_batch_counter = 0
        self.bio_label_list = self._make_bio_label_list()
        self.filter_method = FilterMethod(contrast_filtering_method)

        if self.filter_method == FilterMethod.NO_O:
            # force _make_entity_triplets to not contrast with O-labels
            self.chance_o_contrast = -1.0
        else:
            self.chance_o_contrast = neg_o_prob

        self._label_statistics = {
            "anchors": defaultdict(int),
            "negatives": defaultdict(int)
        }


    @property
    def label_statistics(self) -> Dict[str, Dict[str, int]]:
        return self._label_statistics

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

        log.debug(f"Making label tensor dict")
        labels_dict = self._make_label_tensor_dict(sentences)

        log.debug(f"Filtering labels dict")
        filtered_labels_dict = self._filter_labels_dict(labels_dict, self.filter_method)

        log.debug(f"Making entity triplets")
        final_loss_tensors = self._make_entity_triplets(filtered_labels_dict)

        self._internal_batch_counter += 1

        # calculate the loss
        return self.loss_function(*final_loss_tensors), label_tensor.size(0)

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

        log.debug(f"Filtering labels with method {method}")

        match method:
            case FilterMethod.NONE:
                return labels_dict
            case FilterMethod.NO_O:
                return self._no_o_filter(labels_dict)
            case FilterMethod.LESS_2:
                return self._less_2_filter(labels_dict)
            case FilterMethod.O_ONLY_NEG:
                return labels_dict
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

        log.debug(f"Making triplets for labels: {labels_dict.keys()}")
        assert len(labels_dict) > 0, "No labels found in labels_dict"

        triplets = []

        # here for the case we throw O-labels out or similar
        relevant_labels = [label for label in self.bio_label_list if label in labels_dict.keys()]

        if self.filter_method == FilterMethod.NO_O and "O" in relevant_labels:
            relevant_labels.remove("O")

        log.debug(f"Relevant labels: {relevant_labels}")

        for entity, tensor_list in labels_dict.items():
            # skip if we have no different positive for that label
            if len(tensor_list) < 2:
                log.debug(f"Skipping label {entity} with less than 2 entries")
                continue

            if self.filter_method == FilterMethod.O_ONLY_NEG and entity == "O":
                log.debug(f"Skipping O label for O-only-neg filtering")
                continue

            for current_anchor_idx, current_anchor_tup in enumerate(tensor_list):
                anchor_label = entity
                anchor_tensor = current_anchor_tup[1]

                log.debug(f"Making triplets for label {anchor_label}")

                self._label_statistics["anchors"][anchor_label] += 1

                # contrast with O-labels or other labels
                contrast_with_o = random.random() < self.chance_o_contrast

                if contrast_with_o:
                    possible_neg_labels = ["O"]
                    log.debug(f"Contrasting with O")

                else:
                    possible_neg_labels = [
                        label for label in relevant_labels if (
                            label != anchor_label and
                            len(labels_dict[label]) > 0 and
                            label != "O"
                    )]
                    log.debug(f"Contrasting with {possible_neg_labels}")

                # should be fixed by filtering the dataset now but can never be too safe
                assert len(possible_neg_labels) > 0

                negative_label = choice(possible_neg_labels)
                log.debug(f"Chose negative label {negative_label}")

                self._label_statistics["negatives"][negative_label] += 1

                negative_tensor = choice(labels_dict[negative_label])[1]

                # any entity sharing label with anchor without same index
                available_positives = tensor_list[:current_anchor_idx] + tensor_list[current_anchor_idx + 1:]
                positive_tensor = choice(available_positives)[1]

                triplets.extend(self._make_loss_triplets(anchor_tensor, positive_tensor, negative_tensor))

        log.debug(f"Building full tensors for {len(triplets)} triplets")
        try:
            first_tensor, second_tensor, third_tensor = self._make_torch_stack(triplets)
            a = 1
        except RuntimeError as e:
            log.error(f"Seed {wandb.config.seed} triplet building not possible")
            log.error(f"Error in making triplets: {e}")
            raise e

        return first_tensor, second_tensor, third_tensor

    def _make_loss_triplets(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> List[TRIPLET]:
        if self.loss_used == "tml":
            return [(anchor, positive, negative)]
        elif self.loss_used == "cel":
            return [(anchor, positive, torch.ones(1, device=flair.device)), (anchor, negative, -torch.ones(1, device=flair.device))]
        else:
            raise NotImplementedError(f"Handling {self.loss_function} is not implemented")

    def _make_torch_stack(self, triplets: List[TRIPLET]) -> TRIPLET:
        if self.loss_used == "tml":
            first_tensor = torch.stack([triplet[0] for triplet in triplets])
            second_tensor = torch.stack([triplet[1] for triplet in triplets])
            third_tensor = torch.stack([triplet[2] for triplet in triplets])

            return first_tensor, second_tensor, third_tensor
        elif self.loss_used == "cel":
            first_tensor = torch.stack([triplet[0] for triplet in triplets])
            second_tensor = torch.stack([triplet[1] for triplet in triplets])
            third_tensor = torch.cat([triplet[2] for triplet in triplets])

            return first_tensor, second_tensor, third_tensor
        else:
            raise NotImplementedError(f"Handling {self.loss_function} is not implemented")

    def _make_bio_label_list(self):
        bio_labels = []
        for label in self.label_list:
            if label != "O":
                bio_labels.append(label[2:])
            else:
                bio_labels.append(label)

        return list(set(bio_labels))

