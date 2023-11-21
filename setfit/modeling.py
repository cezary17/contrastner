import logging
import typing
from typing import List, Tuple, Dict

import flair
import torch
from torch.nn import TripletMarginLoss, CrossEntropyLoss
from flair.data import Dictionary, DT, Token, Label
from flair.embeddings import TokenEmbeddings
from flair.models import TokenClassifier

logger = logging.getLogger("flair")
logger.setLevel("DEBUG")

# TODO: Turn this into a proper class this is getting too big imo
LABEL_TENSOR_DICT = Dict[str, List[Tuple[Tuple[int, int, str], torch.Tensor]]]


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
        data_point_tensor = self._encode_data_points(sentences, data_points)

        # HUGE CHANGES HERE!
        scores = self.decoder(data_points=data_points, data_point_tensor=data_point_tensor)

        # an optional masking step (no masking in most cases)
        scores = self._mask_scores(scores, data_points)

        # calculate the loss
        return self._calculate_loss(scores, label_tensor)


class SetFitDecoder(torch.nn.Module):
    """
    1. Erstelle Key-Value / Dict mit Label: Tokens, z.B. {"PER": [token1, token2], ..., "O": [tokenN, ..., tokenX]}
        -> Funktion _make_label_token_dict

    2. Forme Triplets aus diesem Dict. Beginn mit trivialer Idee: Forme für jeden Token im Batch ein Triplet
        z.b. [(anchor: token1, pos: token2, neg: tokenX), ...]. Geht nur wenn mind. 2 Labels pro klasse vorhanden sind,
        da sonst kein positiv anchor. Negative einfach random aus allen anderen Labels.

    3. Dann diese List durch torch.nn.TripletMarginLoss o. ä., können wir noch anpassen aber das ist erstmal der Standard.

    4. Return die Scores (negative distance) aus batch

    5. Train model mit 100 beispiele aus CoNLL
    """
    def __init__(self, label_dictionary: Dictionary, *args, **kwargs):
        """
        who even reads docstrings?

        :param fs_examples: a list of Sentences -> Few Shot Examples
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.label_dictionary = label_dictionary

        label_list = self.label_dictionary.get_items()
        label_list += ["O"]
        self.label_list = label_list

        # Will be set and updated in forward
        self.label_tensor_dict = None
        self.internal_batch_counter = 0

    def forward(self, data_points: list[Token], data_point_tensor):

        label_token_dict = self._make_label_tensor_dict(data_points)
        self._update_internal_label_tensor(label_tensor_dict=label_token_dict)
        error_replacement = False

        # Assert that each label has at least 2 tokens
        for label, tokens in self.label_tensor_dict.items():
            # PROBLEM HERE: THIS ALWAYS HAPPENS WITH BATCH SIZES UP TO 64
            # IDEA -> 2 LOSS FUNCTIONS AND WE INSTEAD KEEP A GLOBAL LABEL-TENSOR DICT WHICH WE UPDATE UP TO THE POINT
            # WHEN WE HAVE ENOUGH DATA -> THEN WE SWITCH TO TRIPLET LOSS
            if len(tokens) < 2:
                # raise ValueError(f"Label {label} has less than 2 tokens. ")
                error_replacement = True

        self.internal_batch_counter += 1
        return data_point_tensor

    def _make_entity_triplets(
            self,
            labels: List[str],
            inputs: List[torch.Tensor] = None,
            k: int = 20) -> List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        This will get the entities from a sentence and return them as a list of tuples (entity, label, index)

        :param inputs:
        :param labels:
        :param k:
        :return:
        """

        # Output format: [(anchor, (positive, negative)), ...]
        pass

    def _make_label_tensor_dict(self, data_points: list[Token]) -> LABEL_TENSOR_DICT:
        """
        This will get the input tensor batch and output a dictionary of the labels alongside their index and the
        corresponding tensors

        :param data_points: list of data points (flair.data.Token) as seen in TokenClassifier forward loop
        :return: A dict mapping the label to tensors having that label alongside it's index eg:
            {
                "LOC": [(0, Tensor1), (1, Tensor2)
                "PER": [(2, Tensor3), (3, Tensor4)
                ...
            }
        """

        # Initialize dict of labels with labels found in label_dictionary
        labels_dict = {label: [] for label in self.label_list}

        for dp_idx, dp in enumerate(data_points):

            dp_label = dp.get_label("ner")

            # Map to simple embeddings to get "LOC" from "S-LOC", ...
            dp_simple_label = self._cut_label_prefix(dp_label)

            labels_dict[dp_simple_label].append(((self.internal_batch_counter, dp_idx, dp.text), dp.embedding))

        return labels_dict

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
        assert label_val in self.label_list

        return label_val

    def _update_internal_label_tensor(self, label_tensor_dict: LABEL_TENSOR_DICT):
        """
        Function for updating the internal label-tensor-dict. Needed as we don't get enough different labels in first
        step(s) to do TripletMarginLoss

        Also, as of now checks if a batch had >= 2 NER labels inside

        :param label_tensor_dict: Output from _make_label_token_dict
        :return:
        """

        feature_rich = all([True if len(val) >= 2 else False for val in label_tensor_dict.values()])

        if feature_rich:
            logger.debug(msg=f"FEATURE RICH BATCH NUMBER {self.internal_batch_counter} LESGO")

        if self.label_tensor_dict is None:
            self.label_tensor_dict = label_tensor_dict
            return

        for key in self.label_tensor_dict:
            self.label_tensor_dict[key] += label_tensor_dict[key]
