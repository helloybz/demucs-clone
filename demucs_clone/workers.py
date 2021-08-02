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

        dataloader = self._init_dataloader()

        for x, y in dataloader:
            self.optimizer.zero_grad()
            y_hat = self.model(x)
            cost = self.criterion(input=y, target=y_hat)
            cost.backward()
            self.optimizer.step()

            yield cost.item()
