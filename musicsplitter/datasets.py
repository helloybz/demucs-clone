import random

import torch
import musdb


class MUSDB18(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        download=False,
        subsets=['train', 'test'],
        chunk_duration_in_sec=5,
        sample_rate=44100
    ):
        super(MUSDB18, self).__init__()
        self.mus = musdb.DB(
            root=data_root,
            download=download,
            subsets=subsets,
            sample_rate=sample_rate
        )
        self.chunk_duration_in_sec = chunk_duration_in_sec
        self.sample_rate = sample_rate

    def __getitem__(self, idx):
        track = self.mus[idx]

        # make random crop
        track.chunk_duration = self.chunk_duration_in_sec
        track.chunk_start = random.uniform(0, track.duration - self.chunk_duration_in_sec)

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
