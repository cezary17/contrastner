import typing
from typing import List, Tuple

import flair
import torch
from flair.data import Dictionary, DT, Token, Label
from flair.embeddings import TokenEmbeddings
from flair.models import TokenClassifier

'''
TODO:
1. Erstelle Key-Value / Dict mit Label: Tokens, z.B. {"PER": [token1, token2], ..., "O": [tokenN, ..., tokenX]}
    -> Funktion _make_label_token_dict

2. Forme Triplets aus diesem Dict. Beginn mit trivialer Idee: Forme für jeden Token im Batch ein Triplet
    z.b. [(anchor: token1, pos: token2, neg: tokenX), ...]. Geht nur wenn mind. 2 Labels pro klasse vorhanden sind, 
    da sonst kein positiv anchor. Negative einfach random aus allen anderen Labels.
    
3. Dann diese List durch torch.nn.TripletMarginLoss o. ä., können wir noch anpassen aber das ist erstmal der Standard.

4. Return die Scores (negative distance) aus batch

5. Train model mit 100 beispiele aus CoNLL
'''


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

    def forward(self, data_points: list[Token], data_point_tensor: typing.Any):

        label_token_dict = self._make_label_token_dict(data_points)
        print("I AM HERE TO HAVE SOMEWHERE TO BREAKPOINT")

        return data_points

    def _make_entity_triplets(
            self,
            labels: list[str],
            inputs: list[torch.Tensor] = None,
            k: int = 20) -> list[tuple[torch.Tensor (torch.Tensor, torch.Tensor)]]:
        """
        This will get the entities from a sentence and return them as a list of tuples (entity, label, index)

        :param inputs:
        :param labels:
        :param k:
        :return:
        """

        # Output format: [(anchor, (positive, negative)), ...]
        pass

    def _make_label_token_dict(self, data_points: list[Token]) -> dict[str, (int, list[torch.Tensor])]:
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
        labels_dict = {label: []
                       for label in self.label_dictionary.get_items()}
        labels_dict["O"] = []

        for dp_idx, dp in enumerate(data_points):

            dp_label = dp.get_label("ner")

            # Map to simple embeddings to get "LOC" from "S-LOC", ...
            dp_simple_label = self._map_pos_labels(dp_label)

            labels_dict[dp_simple_label].append((dp_idx, dp.embedding))

        return labels_dict

    def _map_pos_labels(self, label: Label) -> str:
        """
        Maybe possible that this function is redundant and flair provides the simple label somewhere
        TODO: Ask Jonas

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
