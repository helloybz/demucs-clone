import argparse
import random
from pathlib import Path

import torch
import numpy
import yaml

from .models import Demucs
from .datasets import MUSDB18
from .workers import Trainer
from .workers import Validator
from . import loss


def train(args):
    config_file = Path(args.config_file)
    assert config_file.exists()

    hparams = yaml.load(
        stream=open(config_file),
        Loader=yaml.FullLoader,
    )
    data_root = Path(__file__).parent.parent.parent.joinpath("datasets").joinpath('musdb18')
    assert data_root.exists()

    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    train_dataset = MUSDB18(
        data_root=data_root,
        split='train',
        sources=hparams['sources'],
        sample_rate=hparams['sample_rate'],
        **hparams['dataset']['train'],
    )
    valid_dataset = MUSDB18(
        data_root=data_root,
        split='valid',
        sources=hparams['sources'],
        sample_rate=hparams['sample_rate'],
        **hparams['dataset']['valid'],
    )

    model = Demucs(sample_rate=hparams['sample_rate'], sources=hparams['sources'])
    model.to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        **hparams['optimizer'],
    )

    criterion = getattr(loss, hparams['criterion']['method'])(
        **hparams['criterion']['kwargs'] or {})

    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=hparams['batch_size'],
        num_workers=hparams['num_workers'],
        device=device,
    )
    validator = Validator(
        model=model,
        dataset=valid_dataset,
        criterion=criterion,
        batch_size=hparams['batch_size'],
        num_workers=hparams['num_workers'],
        validation_period=hparams['validation_period'],
        device=device,
    )

    epoch_start = 0
    epoch_end = args.epochs
    for epoch in range(epoch_start, epoch_end):
        loss_train_epoch = 0
        batch_size = 0
        for loss_train_batch in trainer.train():
            print(f'train_batch {loss_train_batch}', end='\r')
            loss_train_epoch += loss_train_batch
            batch_size += 1

        print(f'train_epoch {loss_train_epoch/batch_size}')

        if epoch & validator.validation_period == 0:
            loss_valid_epoch = 0
            batch_size = 0
            for loss_valid_batch in validator.validate():
                print(f'valid_batch {loss_valid_batch}', end='\r')
                loss_valid_epoch += loss_valid_batch
                batch_size += 1
            loss_valid_average = loss_valid_epoch/batch_size
            print(f'valid_epoch {loss_valid_average}')

            if validator.is_best(loss_valid_average):
                # save_model
                pass


def main():
    parser = argparse.ArgumentParser(
        prog="demucs_clone"
    )

    parser.add_argument("--config_file", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gpu", action='store_true')

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(0)
    random.seed(0)
    main()
