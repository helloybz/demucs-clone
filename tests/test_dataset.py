from pathlib import Path

import unittest
import torch

from demucs_clone.datasets import MUSDB18


class TestMUSDB18(unittest.TestCase):
    def setUp(self):
        data_root = Path('./tests/data_sample').absolute()
        assert data_root.exists(), data_root

        self.mus_train = MUSDB18(
            data_root=data_root,
            download=False,
            split='train',
            chunk_duration=5,
            sample_rate=44100,
        )

        self.mus_valid = MUSDB18(
            data_root=data_root,
            download=False,
            split='valid',
            chunk_duration=5,
            sample_rate=44100,
        )

        self.mus_test = MUSDB18(
            data_root=data_root,
            download=False,
            split='test',
            chunk_duration=5,
            sample_rate=44100,
        )

    def test_is_torch_tensor(self):
        data = self.mus_train[0]

        for data_ in data:
            assert isinstance(data_, torch.Tensor)

    def test_length(self):
        assert len(self.mus_train) == 1
        assert len(self.mus_test) == 1

    def test_getitem(self):
        audio, vocal, drum, bass, other = self.mus_train[0]

        assert audio.shape[0] == 2  # stereo
        assert audio.shape[1] == self.mus_train.chunk_duration * self.mus_train.sample_rate

        assert audio.shape == vocal.shape == drum.shape == bass.shape == other.shape

    def test_dtype(self):
        data = self.mus_train[0]

        for data_ in data:
            assert data_.dtype == torch.float32
