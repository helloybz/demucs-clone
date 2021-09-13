import unittest

import torch

from demucs_clone.modules.augmentations import ChannelSwapping, Scaling, SignShifting


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


class TestScaling(unittest.TestCase):
    def setUp(self):
        self.augmentation = Scaling(min_scaler=0.25, max_scaler=1.25)

    def test_output_shape_is_maintained(self):
        B, S, C, T = 2, 4, 2, 10
        signals = torch.rand(B, S, C, T)
        scaled = self.augmentation(signals)

        assert scaled.dim() == 4
        assert scaled.size() == (B, S, C, T)

    def test_scales_along_batch_dims_and_source_dims(self):
        B, S, C, T = 2, 4, 2, 10
        signals = torch.ones(B, S, C, T)
        scaled = self.augmentation(signals)

        # get scalers by indexing all elems in B and S, the first elems in C and T.
        scalers = scaled[:, :, 0, 0].squeeze()

        assert scalers.reshape(B, S, 1, 1).expand(B, S, C, T).eq(scaled).all()

    def test_scales_independantly(self):
        B, S, C, T = 10, 4, 2, 10
        signals = torch.ones(B, S, C, T)
        scaled = self.augmentation(signals)

        # get scalers by indexing all elems in B and S, the first elems in C and T.
        scalers = scaled[:, :, 0, 0].squeeze()
        assert scalers.min() != scalers.max()
