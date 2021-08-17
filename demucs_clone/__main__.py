import argparse
import random
from pathlib import Path

from torch.utils import tensorboard
import torch
import numpy
import yaml

from .modules import Demucs
from .modules.augmentations import ChannelSwapping, Scaling, SourceShuffling
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

    CHECKPOINT_ROOT = Path(__file__).parent.parent.joinpath(f'{config_file.stem}').joinpath('checkpoints')
    tb_logdir = Path(__file__).parent.parent.joinpath(f'{config_file.stem}').joinpath('logdir')

    tb = tensorboard.SummaryWriter(
        log_dir=tb_logdir,
    )

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

    augmentations = [
        ChannelSwapping(prob=0.5),
        Scaling(min_scaler=0.25, max_scaler=1.25),
        SourceShuffling(),
    ]

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        **hparams['optimizer'],
    )

    criterion = getattr(loss, hparams['criterion']['method'])(
        **hparams['criterion']['kwargs'] or {})

    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        augmentations=augmentations,
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

    epoch_start = 1
    epoch_end = args.epochs + 1
    for epoch in range(epoch_start, epoch_end):
        loss_train_epoch = 0
        batch_size = 0
        for loss_train_batch in trainer.train():
            print(f'train_batch {loss_train_batch}', end='\r')
            loss_train_epoch += loss_train_batch
            batch_size += 1

        tb.add_scalar(tag='train', scalar_value=loss_train_epoch/batch_size, global_step=epoch)
        print(f'train_epoch {loss_train_epoch/batch_size}')

        if epoch % validator.validation_period == 0:
            loss_valid_epoch = 0
            batch_size = 0
            metrics = {'sdr': 0, 'isr': 0, 'sir': 0, 'sar': 0}
            num_examples = 2

            for batch_idx, (loss_valid_batch, metrics_batch, example) in enumerate(validator.validate(num_examples=num_examples)):

                print(f'valid_batch {loss_valid_batch}', end='\r')
                batch_size += 1

                if len(example) > 0:
                    mixture, sources = example
                    tb.add_audio(tag=f'mixture/{batch_idx}', snd_tensor=mixture.cpu().squeeze(0).mean(0, keepdim=True), global_step=epoch, sample_rate=hparams['sample_rate'])
                    for soure_name, source in zip(valid_dataset.sources, sources.cpu().squeeze(0)):
                        tb.add_audio(tag=f'{soure_name}/{batch_idx}', snd_tensor=source.mean(0, keepdim=True), global_step=epoch, sample_rate=hparams['sample_rate'])

                loss_valid_epoch += loss_valid_batch
                for metric_key in metrics_batch.keys():
                    metrics[metric_key] += metrics_batch[metric_key]

            loss_valid_average = loss_valid_epoch/batch_size
            metrics_average = {key: value/batch_size for key, value in metrics.keys()}

            tb.add_scalar(tag='valid', scalar_value=loss_valid_average, global_step=epoch)
            tb.add_scalars(
                main_tag='valid',
                tag_scalar_dict=metrics_average,
                global_step=epoch
            )

            print(f'valid_epoch {loss_valid_average}')

            if validator.is_best(loss_valid_average):
                context = trainer.get_context()
                context['epoch'] = epoch
                context['hparams'] = hparams
                torch.save(context, CHECKPOINT_ROOT.joinpath('best'))

        if epoch % args.checkpoint_period == 0:
            context = trainer.get_context()
            context['epoch'] = epoch
            context['hparams'] = hparams
            torch.save(context, CHECKPOINT_ROOT.joinpath(f'{epoch}'))

        tb.flush()


def main():
    parser = argparse.ArgumentParser(
        prog="demucs_clone"
    )

    parser.add_argument("--config_file", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint_period", type=int)
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
