import argparse
import random
from pathlib import Path

from diffq import DiffQuantizer
from torch.utils import tensorboard
import torch
from torch import distributed
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm import tqdm
import multiprocessing as mp
import numpy
import yaml

from .modules import Demucs
from .modules.augmentations import ChannelSwapping, SignShifting, Scaling, SourceShuffling
from .datasets import MUSDB18
from .workers import Trainer
from .workers import Validator
from . import loss


def train(args):
    config_file = Path(args.config_file)
    assert config_file.exists(), f"The given config_file is not exists. {args.config_file}"

    with open(config_file) as config_io:
        hparams = yaml.load(
            stream=config_io,
            Loader=yaml.FullLoader,
        )
    data_root = Path(args.data_root)
    assert data_root.exists()

    if args.rank == 0:
        CHECKPOINT_ROOT = Path(args.checkpoint_root)
        tb_logdir = Path(CHECKPOINT_ROOT).parent.joinpath('logdir' if not args.is_testing else 'logdir_test')

        CHECKPOINT_ROOT.mkdir(exist_ok=True)

    if args.rank == 0:
        tb = tensorboard.SummaryWriter(
            log_dir=tb_logdir,
        )

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.rank}')
        torch.cuda.set_device(device=device)
        if args.world_size > 1:
            distributed.init_process_group(
                backend='nccl',
                init_method=f'tcp://{args.master_url}',
                world_size=args.world_size,
                rank=args.rank,
            )
    else:
        device = torch.device('cpu')

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

    model = Demucs(
        sample_rate=hparams['sample_rate'],
        sources=hparams['sources'],
        **hparams['model'],
    )

    if torch.cuda.is_available():
        model.to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        **hparams['optimizer'],
    )
    quantizer = DiffQuantizer(model=model, group_size=8)
    quantizer.setup_optimizer(optimizer=optimizer)
    augmentations = [
        ChannelSwapping(prob=0.5),
        SignShifting(prob=0.5),
        Scaling(min_scaler=0.25, max_scaler=1.25),
        SourceShuffling(),
    ]
    augmentations = torch.nn.Sequential(*augmentations)
    augmentations.to(device)

    # Wrapping with DDP should be done after init the quantizer.
    if args.world_size > 1:
        dmodel = DistributedDataParallel(
            module=model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
        )
    else:
        dmodel = model

    criterion = getattr(loss, hparams['criterion']['method'])(
        **hparams['criterion']['kwargs'] or {})

    trainer = Trainer(
        model=dmodel,
        dataset=train_dataset,
        augmentations=augmentations,
        criterion=criterion,
        optimizer=optimizer,
        quantizer=quantizer,
        quantizer_penalty=hparams['quantizer']['penalty'],
        batch_size=hparams['batch_size'],
        num_workers=args.num_workers,
        device=device,
        world_size=args.world_size,
    )
    validator = Validator(
        model=model,
        dataset=valid_dataset,
        criterion=criterion,
        batch_size=hparams['batch_size'],
        num_workers=args.num_workers,
        device=device,
        world_size=args.world_size,
    )

    epoch_start = 1
    epoch_end = args.epochs + 1
    for epoch in range(epoch_start, epoch_end):

        # Training
        loss_train_epoch = 0
        num_iters = len(train_dataset) // hparams['batch_size']

        if not args.is_testing:
            for loss_train_batch in tqdm(trainer.train(epoch=epoch), desc=f'Train | Epoch {epoch}', total=num_iters):
                loss_train_epoch += loss_train_batch

        if args.rank == 0:
            tb.add_scalar(tag='train', scalar_value=loss_train_epoch/num_iters, global_step=epoch)
            print(f'EPOCH:{epoch}\tAvg Train Loss:{loss_train_epoch/num_iters:.4f}')

        # Validation
        loss_valid_epoch = 0
        num_iters = len(valid_dataset) // validator.dataloader.batch_size

        if not args.is_testing:
            for loss_valid_batch in tqdm(validator.validate(epoch), desc=f'Valid | Epoch {epoch}', total=len(valid_dataset)):
                loss_valid_epoch += loss_valid_batch

        loss_valid_average = loss_valid_epoch / len(valid_dataset)

        if args.rank == 0:
            tb.add_scalar(tag='valid', scalar_value=loss_valid_average, global_step=epoch)
            print(f'EPOCH:{epoch}\tAvg Valid Loss:{loss_valid_average:.4f}')

            context = trainer.get_context()
            context['epoch'] = epoch
            context['hparams'] = hparams
            torch.save(context, CHECKPOINT_ROOT.joinpath(f'epoch_{epoch:04d}.pt'))

            if validator.is_best(loss_valid_average):
                torch.save(context, CHECKPOINT_ROOT.joinpath('best.pt'))

    tb.flush()


def main():
    parser = argparse.ArgumentParser(
        prog="demucs_clone"
    )

    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--config_file", type=Path, required=True)
    parser.add_argument("--checkpoint_root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--master_url", type=str, default='')
    parser.add_argument("--is_testing", action='store_true', default=False)

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
    mp.set_start_method('forkserver')
    main()
