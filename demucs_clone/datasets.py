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

        self._segment_map = [int(track.duration) - self.chunk_duration + 1 for track in self.mus]

    def __getitem__(self, idx):

        cursor = 0
        for i, segment_size in enumerate(self._segment_map):
            next_cursor_position = cursor + segment_size
            if next_cursor_position > idx:
                track_idx = i
                offset = idx - cursor
                break
            else:
                cursor = next_cursor_position

        track = self.mus[track_idx]

        # make random crop
        track.chunk_duration = self.chunk_duration - 1
        track.chunk_start = offset + random.uniform(0, 1)

        mixture = torch.from_numpy(track.audio.T).float()

        sources = []
        for source in self.sources:
            target_source_audio = torch.from_numpy(track.sources[source].audio.T).float()
            sources.append(target_source_audio)
        sources = torch.stack(sources)

        return mixture, sources

    def __len__(self):
        return sum(self._segment_map)
