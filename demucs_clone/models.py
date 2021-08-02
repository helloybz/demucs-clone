from typing import Sequence
import torch.nn as nn
import torchaudio.functional as F_audio

from .utils import init_conv_weight_with_rescaling
from .utils import trim_edge


class Demucs(nn.Module):
    """
    Takes a raw waveform.
    Return an isolated instrument in a waveform.
    """

    def __init__(
        self,
        sample_rate:    int = 44100,
        sources:        Sequence[str] = ['drums', 'bass', 'vocals', 'other'],
    ) -> None:
        super(Demucs, self).__init__()

        self.sample_rate = sample_rate
        self.sources = sources
        self.encoder_conv_blocks = nn.ModuleList(
            [
                DemucsEncoderBlock(
                    in_channels=2,
                    out_channels=64,
                ),
                DemucsEncoderBlock(
                    in_channels=64,
                    out_channels=128,
                ),
                DemucsEncoderBlock(
                    in_channels=128,
                    out_channels=256,
                ),
                DemucsEncoderBlock(
                    in_channels=256,
                    out_channels=512,
                ),
                DemucsEncoderBlock(
                    in_channels=512,
                    out_channels=1024,
                ),
                DemucsEncoderBlock(
                    in_channels=1024,
                    out_channels=2048,
                ),
            ]
        )
        self.encoder_bilstm = nn.LSTM(
            input_size=2048,
            hidden_size=2048,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.encoder_linear = nn.Linear(
            in_features=4096,
            out_features=2048,
        )

        self.decoder_conv_blocks = nn.ModuleList(
            [
                DemucsDecoderBlock(
                    in_channels=2048,
                    out_channels=1024,
                ),
                DemucsDecoderBlock(
                    in_channels=1024,
                    out_channels=512,
                ),
                DemucsDecoderBlock(
                    in_channels=512,
                    out_channels=256,
                ),
                DemucsDecoderBlock(
                    in_channels=256,
                    out_channels=128,
                ),
                DemucsDecoderBlock(
                    in_channels=128,
                    out_channels=64,
                ),
                DemucsDecoderBlock(
                    in_channels=64,
                    out_channels=2 * len(self.sources),  # 2 channels for each 4 instruments.
                ),
            ]
        )

        # Remove the activation function from the last decoder block.
        self.decoder_conv_blocks[-1].block = self.decoder_conv_blocks[-1].block[:-1]

        self.apply(init_conv_weight_with_rescaling)

    def forward(self, x):
        B, C, T = x.shape
        x = F_audio.resample(
            waveform=x,
            orig_freq=self.sample_rate,
            new_freq=self.sample_rate*2,
        )

        encoder_outputs = []
        for encoder_conv_block in self.encoder_conv_blocks:
            x = encoder_conv_block(x)
            encoder_outputs.append(x)
        encoder_outputs.reverse()

        x = x.transpose(-1, -2)  # C becomes d, and (B, d, T) to (B, T, d)
        x, _ = self.encoder_bilstm(x)
        x = self.encoder_linear(x)
        x = x.transpose(-1, -2)

        for encoder_output, decoder_conv_block in zip(encoder_outputs, self.decoder_conv_blocks):
            encoder_output = trim_edge(input=encoder_output, target=x)
            x = encoder_output.add(x)
            x = decoder_conv_block(x)

        x = F_audio.resample(
            waveform=x,
            orig_freq=self.sample_rate*2,
            new_freq=self.sample_rate,
        )
        x = x.reshape(B, len(self.sources), C, -1)
        return x

    def _trim_edge(self, encoder_output, decoder_input):
        diff = encoder_output.shape[-1] - decoder_input.shape[-1]
        assert diff >= 0
        encoder_output = encoder_output[..., diff//2:encoder_output.shape[-1]-(diff-diff//2)]

        return encoder_output


class DemucsEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DemucsEncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=2*out_channels,
                kernel_size=1,
                stride=1,
            ),
            nn.GLU(dim=-2),  # halves along C.
        )

    def forward(self, x):
        return self.block(x)


class DemucsDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DemucsDecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(  # supposed to decrease T. Figure2b in the paper.
                in_channels=in_channels,
                out_channels=2*in_channels,
                kernel_size=3,
                stride=1,
            ),
            nn.GLU(dim=-2),  # halves along C.
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)
