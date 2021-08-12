"""
Data augmentations, which are supposed to be applied to the one batch, are implemented here.
"""

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
            signals: torch.Tensor,
    ) -> torch.Tensor:
        '''
            signasls: [torch.Tensor(*, B, C=2, T)]
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
        signals: torch.Tensor,
    ) -> torch.Tensor:

        scaler = self.uniform_dist.sample([1]).to(signals.device)
        for i, signal in enumerate(signals):
            signals[i] = signal*scaler

        return signals


class SourceShuffling(nn.Module):
    def __init__(self) -> None:
        super(SourceShuffling, self).__init__()

    def forward(
        self,
        signals: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Shuffles the sources along B.
        signals: (B, S, C, T)
            B: Batch size
            S: Number of sources
            C: Number of channels
            T: Number of samples
        '''
        assert signals.dim() == 4

        B, S, C, T = signals.size()
        index = torch.stack(
            [torch.randperm(B, device=signals.device) for s in range(S)],
            dim=1
        )
        # reshpe index as the shape of the signals.
        index = index.reshape(*index.size(), *[1]*(signals.dim()-index.dim()))

        signals = signals.gather(dim=0, index=index.expand(signals.size()))
        del index

        return signals
