from pathlib import Path
import unittest

import torch

from demucs_clone.models import Demucs
from demucs_clone.datasets import MUSDB18


class TestDemucs(unittest.TestCase):
    def setUp(self):
        self.model = Demucs(sample_rate=44100)

        data_root = Path('./tests/data_sample').absolute()
        assert data_root.exists(), data_root

        self.mus_train = MUSDB18(
            data_root=data_root,
            download=False,
            split='train',
            chunk_duration=2,
            sample_rate=44100,
        )

    def test_model_forward(self):
        with torch.no_grad():
            for x, *y in torch.utils.data.DataLoader(self.mus_train, batch_size=1, shuffle=True):
                y_hat = self.model(x)
                assert y_hat.shape[-1] == 81238
                break

    def test_model_grad(self):
        # Search how to test this.
        pass

    def test_model_weight_init(self):
        # TODO Write test code:
        pass
