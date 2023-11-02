import torch


class SetFitDecoder(torch.nn.Module):

    def __init__(self, fs_examples, *args, **kwargs):
        """
        who even reads docstrings?

        :param fs_examples: the given few shot examples
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        raise NotImplementedError #TODO: Implement SetFit

    def forward(self, inputs: torch.Tensor):
        raise NotImplementedError #TODO: Implement SetFit