import unittest

from demucs_clone.modules import Demucs


class TestModel(unittest.TestCase):
    def test_channel_size(self):
        INITIAL_CHANNEL = 4
        NUM_LAYERS = 6
        model = Demucs(num_layers=NUM_LAYERS, initial_channel=INITIAL_CHANNEL)

        channels = [2] + [INITIAL_CHANNEL * 2**layer_idx for layer_idx in range(NUM_LAYERS)]

        encoder_channels = [(getattr(getattr(block, 'block')[0], "in_channels")) for block in model.encoder_conv_blocks]
        decoder_channels = [(getattr(getattr(block, 'block')[0], "in_channels")) for block in model.decoder_conv_blocks]
        self.assertEqual(encoder_channels, channels[:-1])
        self.assertEqual(decoder_channels, list(reversed(channels))[:-1])
