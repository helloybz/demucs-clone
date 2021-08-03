from typing import Iterator, Iterable

from numpy import Inf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Worker:
    def __init__(
        self,
        model: nn.Module,
        dataset,
        criterion,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _init_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )


class Trainer(Worker):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        criterion,
        optimizer,
        batch_size,
        num_workers,
    ) -> None:

        super(Trainer, self).__init__(model, dataset, criterion, batch_size, num_workers)

        self.optimizer = optimizer

    def train(self) -> None:
        '''
        Train the model with dataset for 1 epoch.
        '''
        self.model.train()
        dataloader = self._init_dataloader()

        for x, y in dataloader:
            self.optimizer.zero_grad()
            y_hat = self.model(x)
            cost = self.criterion(input=y, target=y_hat)
            cost.backward()
            self.optimizer.step()

            yield cost.item()


class Validator(Worker):
    def __init__(
        self,
        model:              nn.Module,
        dataset:            Iterable,
        criterion:          nn.Module,
        batch_size:         int,
        num_workers:        int,
        validation_period:  int,
    ) -> None:
        super(Validator, self).__init__(model, dataset, criterion, batch_size, num_workers)

        self.validation_period = validation_period
        self.loss_best = Inf

    @torch.no_grad()
    def validate(self) -> Iterator[float]:
        self.model.eval()
        dataloader = self._init_dataloader()

        for x, y in dataloader:
            y_hat = self.model(x)
            cost = self.criterion(input=y, target=y_hat)

            yield cost.item()

    def is_best(self, current_loss: float):
        if self.loss_best > current_loss:
            self.loss_best = current_loss
            return True
        return False
