import unittest

import torch
from demucs_clone.models import Demucs


class TestDemucs(unittest.TestCase):
    def setUp(self):
        self.model = Demucs()

    def test_model_forward(self):
        B, C, T = 4, 2, 44100
        audio = torch.rand(B, C, T)

        with torch.no_grad():
            output = self.model(audio)

        assert output.shape == (B, 2*4, 31404)

    def test_model_grad(self):
        # Search how to test this.
        pass
