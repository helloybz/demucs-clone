"""
Data augmentations, which are supposed to be applied to the one batch, are implemented here.
"""
from typing import Sequence

import torch.nn as nn
import torch


class ChannelSwapping(nn.Module):
    def __init__(
            self,
            prob: float = 0.5,
    ) -> None:
        super(ChannelSwapping, self).__init__()
        self.prob = prob

    def forward(
            self,
            signals: Sequence[torch.Tensor],
    ) -> Sequence[torch.Tensor]:
        '''
            signals: [torch.Tensor(*, B, C=2, T)]
        '''
        flag = torch.bernoulli(torch.Tensor([self.prob]))
        if flag:
            for i, signal in enumerate(signals):
                *_, C, T = signal.shape
                assert C == 2, f"The number of the channel should be 2, but given is {C}"
                signals[i] = signal.flip(-2)  # Flip over C-dim.
        return signals


class Scaling(nn.Module):
    def __init__(
            self,
            min_scaler,
            max_scaler,
    ) -> None:
        super(Scaling, self).__init__()
        self.uniform_dist = torch.distributions.Uniform(low=min_scaler, high=max_scaler)

    def forward(
        self,
        signals: Sequence[torch.Tensor],
    ) -> Sequence[torch.Tensor]:

        scaler = self.uniform_dist.sample([1])
        for i, signal in enumerate(signals):
            signals[i] = signal*scaler

        return signals
