# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
from argparse import ArgumentParser

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer

from data import DInterface
from model import MInterface
from utils import load_model_path_by_args


def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_acc',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))

    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.resume_from_checkpoint = load_path

    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir=args.logdir_path, name=args.log_dir)
    # args.callbacks = load_callbacks()
    # args.logger = logger

    trainer = Trainer.from_argparse_args(args, gpus=2, strategy='ddp', precision=16, limit_test_batches=0.01)
    # trainer = Trainer.from_argparse_args(args, gpus=2, strategy='ddp', fast_dev_run=True)

    trainer.fit(model, data_module, )


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Infoe

    parser.add_argument('--dataset', default='image_dataset', type=str)
    parser.add_argument('--data_dir', default='/home/pyz/data/COCO', type=str)
    parser.add_argument('--model_name', default='model_image_distilled', type=str)
    parser.add_argument('--loss', default=['kl', 'l1'], nargs='+', type=list)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)

    # Model Hyperparameters
    parser.add_argument('--input_resolution', default=224, type=int)
    parser.add_argument('--patch_size', default=32, type=int)
    parser.add_argument('--width', default=384, type=int)
    parser.add_argument('--t', default=4, type=int)
    parser.add_argument('--weight', default=[0.5, 0.5], nargs='+', type=list)
    parser.add_argument('--loss_scale', default=[10, 1], nargs='+', type=list)
    parser.add_argument('--layers', default=6, type=int)
    parser.add_argument('--heads', default=12, type=int)
    parser.add_argument('--output_dim', default=512, type=int)
    parser.add_argument('--teacher_name', default='ViT-B/32', type=str)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    ## Deprecated, old version
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100)

    args = parser.parse_args()

    # List Arguments
    args.mean_sen = [0.485, 0.456, 0.406]
    args.std_sen = [0.229, 0.224, 0.225]

    main(args)
