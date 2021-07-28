from pathlib import Path

import unittest
import torch

from musicsplitter.datasets import MUSDB18


class TestMUSDB18(unittest.TestCase):
    def setUp(self):
        data_root = Path('./tests/data_sample').absolute()
        assert data_root.exists(), data_root

        self.mus_train = MUSDB18(
            data_root=data_root,
            download=False,
            subsets='train',
            chunk_duration_in_sec=10,
            sample_rate=22050,
        )

    def test_is_torch_tensor(self):
        data = self.mus_train[0]

        for data_ in data:
            assert isinstance(data_, torch.Tensor)

    def test_length(self):
        assert len(self.mus_train) == 1

    def test_getitem(self):
        audio, vocal, drum, bass, other = self.mus_train[0]

        assert audio.shape[0] == 2  # stereo
        assert audio.shape[1] == self.mus_train.chunk_duration_in_sec * self.mus_train.sample_rate

        assert audio.shape == vocal.shape == drum.shape == bass.shape == other.shape
