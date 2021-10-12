import sys
sys.path.append('.')
sys.path.append('..')

from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark = False
from torch.utils.data import DataLoader

# setup locaml
from patchmatch_rl import PatchMatchRL
import data_modules as DataModules

def main(args, hparams):
    experiment_name = args.experiment_name

    # setup checkpoint loading
    checkpoint_path = Path(args.checkpoint_path) / experiment_name
    data_path = Path(args.dataset_path)
    log_path = Path(args.log_path)

    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor='val/percent_inlier_2_theta', save_top_k=-1, save_last=True)
    logger = TensorBoardLogger(log_path, name=experiment_name, version=0)

    last_ckpt = checkpoint_path / 'last.ckpt' if args.resume else None
    if (last_ckpt is None) or (not (last_ckpt.exists())):
        last_ckpt = None
    else:
        last_ckpt = str(last_ckpt)

    # setup model / trainer
    model = PatchMatchRL(hparams)
    trainer = Trainer.from_argparse_args(args,
                                         resume_from_checkpoint=last_ckpt,
                                         logger=logger, 
                                         flush_logs_every_n_steps=1,
                                         callbacks=[checkpoint_callback],
                                         log_every_n_steps=1,
                                         weights_summary="full")

    # setup data module
    data_module = getattr(DataModules, args.dataset)(data_path, num_views=hparams.num_views, options=vars(args))
    data_module.setup('fit')

    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = PatchMatchRL.add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()

    # add PROGRAM level args
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dataset', type=str, choices=DataModules.__all__, required=True)
    parser.add_argument('--resume', dest='resume', action='store_true')

    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(resume=False)

    args = parser.parse_args()

    main(args, hparams)
