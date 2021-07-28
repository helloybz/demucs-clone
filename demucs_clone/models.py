import torch.nn as nn


class Demucs(nn.Module):
    """
    Takes a raw waveform.
    Return an isolated instrument in a waveform.
    """

    def __init__(self):
        super(Demucs, self).__init__()
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

    def forward(self, x):
        for encoder_conv_block in self.encoder_conv_blocks:
            x = encoder_conv_block(x)
        return x


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
