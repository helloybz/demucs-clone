from pathlib import Path

import unittest
import torch

from demucs_clone.datasets import MUSDB18


class TestMUSDB18(unittest.TestCase):
    def setUp(self):
        self.data_root = Path('./tests/data_sample').absolute()
        assert self.data_root.exists(), self.data_root

        self.mus_train = MUSDB18(
            data_root=self.data_root,
            download=False,
            split='train',
            chunk_duration=5,
            sample_rate=44100,
        )

        self.mus_valid = MUSDB18(
            data_root=self.data_root,
            download=False,
            split='valid',
            chunk_duration=5,
            sample_rate=44100,
        )

        self.mus_test = MUSDB18(
            data_root=self.data_root,
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
        audio, targets = self.mus_train[0]

        assert audio.shape[0] == 2  # stereo
        assert audio.shape[1] == self.mus_train.chunk_duration * self.mus_train.sample_rate

        assert audio.shape[-1] == targets.shape[-1]

    def test_dtype(self):
        data = self.mus_train[0]

        for data_ in data:
            assert data_.dtype == torch.float32

    def test_select_sources_from_dataset(self):
        C = 2
        sources = ['drums', 'bass']
        mus_drum_bass = MUSDB18(
            data_root=self.data_root,
            download=False,
            split='train',
            sources=sources,
            chunk_duration=5,
            sample_rate=44100,
        )

        assert mus_drum_bass[0][1].shape == (len(sources), C, 44100*5)

        sources = ['drums', 'bass', 'vocals', 'other']
        mus_drum_bass = MUSDB18(
            data_root=self.data_root,
            download=False,
            split='train',
            sources=sources,
            chunk_duration=5,
            sample_rate=44100,
        )

        assert mus_drum_bass[0][1].shape == (len(sources), C, 44100*5)
