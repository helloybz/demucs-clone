from typing import Dict, Iterator, Iterable, Sequence

from numpy import Inf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .datasets import collate_shortest


class Worker:
    def __init__(
        self,
        model: nn.Module,
        dataset,
        criterion,
        batch_size: int,
        num_workers: int,
        device,
        world_size,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.num_workers = num_workers
        self.device = device
        self.world_size = world_size
        self.batch_size = batch_size // world_size

        if world_size > 1:
            self.sampler = DistributedSampler(
                dataset=dataset,
            )

    def _init_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size if self.dataset.split == 'train' else 1,
            shuffle=self.dataset.split == 'train' and (self.sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.dataset.split != 'train',
            collate_fn=collate_shortest,
            sampler=self.sampler if self.world_size > 1 else None,
        )


class Trainer(Worker):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        augmentations:  Sequence[nn.Module],
        criterion,
        optimizer,
        batch_size,
        num_workers,
        device,
        world_size,
    ) -> None:

        super(Trainer, self).__init__(model, dataset, criterion, batch_size, num_workers, device, world_size)

        self.optimizer = optimizer
        self.augmentations = nn.Sequential(*augmentations)
        self.augmentations.to(self.device)

    def train(self, epoch) -> None:
        '''
        Train the model with dataset for 1 epoch.
        '''
        self.model.train()
        dataloader = self._init_dataloader()

        if self.world_size > 1:
            self.sampler.set_epoch(epoch)

        for y in dataloader:
            self.optimizer.zero_grad()
            y = y.to(self.device, non_blocking=True)

            y = self.augmentations(y)
            x = y.sum(1)

            y_hat = self.model(x)
            cost = self.criterion(input=y, target=y_hat)
            cost.backward()
            self.optimizer.step()

            del x, y, y_hat

            yield cost.item()

    def get_context(self) -> Dict:
        context = dict()
        context = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return context


class Validator(Worker):
    def __init__(
        self,
        model:              nn.Module,
        dataset:            Iterable,
        criterion:          nn.Module,
        batch_size:         int,
        num_workers:        int,
        validation_period:  int,
        device,
        world_size,
    ) -> None:
        super(Validator, self).__init__(model, dataset, criterion, batch_size, num_workers, device, world_size)

        self.validation_period = validation_period
        self.loss_best = Inf

    @torch.no_grad()
    def validate(self, epoch, num_examples=2) -> Iterator[float]:
        self.model.eval()
        dataloader = self._init_dataloader()

        if self.world_size > 1:
            self.sampler.set_epoch(epoch)

        for idx, y in enumerate(dataloader):
            y = y.to(self.device, non_blocking=True)
            x = y.sum(1)
            y_hat = self.model(x)
            cost = self.criterion(input=y, target=y_hat)

            if idx < num_examples:
                yield cost.item(), (y_hat.sum(1), y_hat)
            else:
                yield cost.item(), []

    def is_best(self, current_loss: float):
        if self.loss_best > current_loss:
            self.loss_best = current_loss
            return True
        return False
