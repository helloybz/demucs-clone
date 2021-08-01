import random
from typing import Literal

import torch
import musdb


class MUSDB18(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        download:       bool = False,
        split:          Literal['train', 'valid', 'test'] = 'train',
        chunk_duration: int = 5,
        sample_rate:    int = 44100,
    ) -> None:

        super(MUSDB18, self).__init__()

        assert split in ['train', 'valid', 'test']

        if split in ['train', 'valid']:
            self.mus = musdb.DB(
                root=data_root,
                download=download,
                subsets='train',
                split=split,
            )
        elif split in ['test']:
            self.mus = musdb.DB(
                root=data_root,
                download=download,
                subsets='test',
            )
        else:
            raise ValueError

        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate

    def __getitem__(self, idx):
        track = self.mus[idx]

        # make random crop
        track.chunk_duration = self.chunk_duration
        track.chunk_start = random.uniform(0, track.duration - self.chunk_duration)

        audio = track.audio.T
        vocal = track.targets['vocals'].audio.T
        drum = track.targets['drums'].audio.T
        bass = track.targets['bass'].audio.T
        other = track.targets['other'].audio.T

        audio = torch.from_numpy(audio)
        vocal = torch.from_numpy(vocal)
        drum = torch.from_numpy(drum)
        bass = torch.from_numpy(bass)
        other = torch.from_numpy(other)

        return audio, vocal, drum, bass, other

    def __len__(self):
        return len(self.mus.tracks)
