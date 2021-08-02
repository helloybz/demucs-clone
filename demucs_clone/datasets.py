import random
from typing import Literal, Sequence

import torch
import musdb


class MUSDB18(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        download:       bool = False,
        split:          Literal['train', 'valid', 'test'] = 'train',
        sources:        Sequence[str] = ['drums', 'bass', 'vocals', 'other'],
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

        sources.sort()
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.sources = sources

    def __getitem__(self, idx):
        track = self.mus[idx]

        # make random crop
        track.chunk_duration = self.chunk_duration
        track.chunk_start = random.uniform(0, track.duration - self.chunk_duration)

        audio = torch.from_numpy(track.audio.T).float()

        targets = []
        for source in self.sources:
            target_source_audio = torch.from_numpy(track.targets[source].audio.T).float()
            targets.append(target_source_audio)
        targets = torch.stack(targets)

        return audio, targets

    def __len__(self):
        return len(self.mus.tracks)
