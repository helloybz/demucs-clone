import unittest

import torch

from demucs_clone.modules.augmentations import ChannelSwapping, SignShifting


class TestChannelSwapping(unittest.TestCase):
    def setUp(self):
        self.augmentation = ChannelSwapping(prob=1.0)

    def test_swapping(self):
        B, S, C, T = 2, 4, 2, 10
        signals = torch.rand(B, S, C, T)
        swapped = self.augmentation(signals)
        assert (swapped.add(signals)[:, :, 0, :] - swapped.add(signals)[:, :, 1, :]).sum() == 0


class TestSignShifting(unittest.TestCase):
    def setUp(self):
        self.augmentation = SignShifting(prob=1.0)

    def test_shifting(self):
        B, S, C, T = 2, 4, 2, 10
        signals = torch.rand(B, S, C, T)
        shifted = self.augmentation(signals)
        assert shifted.add(signals).sum() == 0
