import random
from typing import Literal, Sequence

import torch
import torchaudio
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
        setup_file:     str = None,
    ) -> None:

        super(MUSDB18, self).__init__()

        assert split in ['train', 'valid', 'test']

        if split in ['train', 'valid']:
            self.mus = musdb.DB(
                root=data_root,
                download=download,
                subsets='train',
                split=split,
                setup_file=setup_file,
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
        self.split = split
        self._segment_map = [int(track.duration) - self.chunk_duration + 1 for track in self.mus]

    def __getitem__(self, idx):

        if self.split == 'train':
            cursor = 0
            for i, segment_size in enumerate(self._segment_map):
                next_cursor_position = cursor + segment_size
                if next_cursor_position > idx:
                    track_idx = i
                    offset = idx - cursor
                    break
                else:
                    cursor = next_cursor_position
        elif self.split in ['valid', 'test']:
            track_idx = idx
        else:
            raise ValueError

        track = self.mus[track_idx]

        if self.split == 'train':
            # make random crop
            track.chunk_duration = self.chunk_duration - 1
            track.chunk_start = offset + random.uniform(0, 1)
        elif self.split == 'valid':
            track.chunk_duration = self.chunk_duration
            track.chunk_start = 0
        else:
            raise ValueError

        sources = []
        for source in self.sources:
            target_source_audio = torch.from_numpy(track.sources[source].audio.T).float()
            sources.append(target_source_audio)

        # Data Augumentation
        # Pitch shift
        if self.split == 'train':
            if torch.bernoulli(torch.Tensor([0.2])):
                effects = [
                    # convert cents into semitones by multiplying 100
                    ['pitch', f'{random.randint(-2, 2)*100}'],
                    ['rate', f'{self.sample_rate}'],
                    ['stretch', f'{random.randint(88, 112)*0.01}'],
                    ['rate', f'{self.sample_rate}'],
                ]

                # TODO: replace with torchaudio.functional.pitch_shift
                for i, source in enumerate(sources):
                    source, _ = torchaudio.sox_effects.apply_effects_tensor(
                        tensor=source,
                        sample_rate=self.sample_rate,
                        effects=effects,
                    )
                    sources[i] = source

        sources = torch.stack(sources)

        return sources

    def __len__(self):
        if self.split == 'train':
            return sum(self._segment_map)
        elif self.split == 'valid':
            return len(self.mus.tracks)
        elif self.split == 'test':
            pass
        else:
            raise ValueError


def collate_shortest(batch_sources):

    shortest_length = min([item.shape[-1] for item in batch_sources])

    sources = [source[..., :shortest_length] for source in batch_sources]

    sources = torch.stack(sources)

    return sources
