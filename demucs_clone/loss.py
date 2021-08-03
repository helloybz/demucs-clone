import torch

import torch.nn.functional as F
from .utils import trim_edge


class L1LossWithTrimming(torch.nn.modules.loss._Loss):

    __constants__ = ['weight', 'pos_weight', 'reduction']

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction='sum',
    ) -> None:
        super(L1LossWithTrimming, self).__init__(
            size_average,
            reduce,
            reduction
        )

    def forward(self, input, target) -> torch.Tensor:
        trimmed_input = trim_edge(input, target)
        return F.l1_loss(trimmed_input, target)
