import argparse
from pathlib import Path
import unittest

import torch

import demucs_clone.__main__ as demucs_clone_main


class TestCheckpointState(unittest.TestCase):
    def setUp(self):
        parser_for_test = argparse.ArgumentParser()
        parser_for_test.add_argument("--data_root", type=Path, required=True)
        parser_for_test.add_argument("--config_file", type=Path, required=True)
        parser_for_test.add_argument("--checkpoint_root", type=Path, required=True)
        parser_for_test.add_argument("--epochs", type=int, required=True)
        parser_for_test.add_argument("--num_workers", type=int, default=0)
        parser_for_test.add_argument("--rank", type=int, default=0)
        parser_for_test.add_argument("--world_size", type=int, default=1)
        parser_for_test.add_argument("--master_url", type=str, default='')
        parser_for_test.add_argument("--is_testing", action='store_true', default=False)

        args = ['--data_root', './tests/data_root', '--config_file', './tests/config_test.yml', '--checkpoint_root', './checkpoints_test', '--epochs', '1', '--is_testing']
        self.args = parser_for_test.parse_args(args)

        demucs_clone_main.train(self.args)

    def test_make_checkpoint_root_if_given_root_is_not_exists(self):
        self.assertTrue(Path('./checkpoints_test').exists())

    def test_checkpoints_is_written(self):
        self.assertTrue(Path('./checkpoints_test').joinpath('epoch_0001.pt').exists())
        checkpoint = torch.load(Path('./checkpoints_test').joinpath('best.pt'))
        self.assertTrue(isinstance(checkpoint, dict))
        self.assertTrue('model' in checkpoint.keys())
        self.assertTrue('optimizer' in checkpoint.keys())
        self.assertTrue('hparams' in checkpoint.keys())
        self.assertTrue('epoch' in checkpoint.keys())

    def test_best_is_written(self):
        self.assertTrue(Path('./checkpoints_test').joinpath('best.pt').exists())

        best_checkpoint = torch.load(Path('./checkpoints_test').joinpath('best.pt'))
        self.assertTrue(isinstance(best_checkpoint, dict))
        self.assertTrue('model' in best_checkpoint.keys())
        self.assertTrue('optimizer' in best_checkpoint.keys())
        self.assertTrue('hparams' in best_checkpoint.keys())
        self.assertTrue('epoch' in best_checkpoint.keys())

    def tearDown(self):
        import shutil

        shutil.rmtree('./logdir_test', ignore_errors=True)
        shutil.rmtree('./checkpoints_test', ignore_errors=True)

        del shutil
