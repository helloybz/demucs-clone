"""
Data augmentations, which are supposed to be applied to the one batch, are implemented here.
"""

import torch.nn as nn
import torch


class ChannelSwapping(nn.Module):
    """
        Apply 'Channel Swapping' to each source in a given minibatch of audio signals.
    """

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
            signals: torch.Tensor  (B, S, C, T)
        '''
        random_channel_idx = torch.bernoulli(torch.full(signals.shape[:-2], self.prob, device=signals.device)).long()
        random_channel_idx = torch.stack([random_channel_idx, 1-random_channel_idx], dim=-1)
        return signals.scatter(dim=-2, index=random_channel_idx.unsqueeze(-1).expand(signals.size()), src=signals)


class SignShifting(nn.Module):
    """
        Apply 'Sign Shifting' to each source in a given minibatch of audio signals.
    """

    def __init__(
        self,
        prob: float = 0.5,
    ) -> None:
        super(SignShifting, self).__init__()
        self.prob = prob

    def forward(
        self,
        signals: torch.Tensor,
    ) -> torch.Tensor:
        B, S, C, T = signals.size()
        random_signs = torch.rand([B, S], device=signals.device).ge(self.prob).int().mul(2).sub(1).reshape(B, S, 1, 1).expand(B, S, C, T)
        return signals * random_signs


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
