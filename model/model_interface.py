# Copyright 2021 Zhongyang Zhang
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

import importlib
import inspect

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lrs
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from .utils import add_similarity, imageQuery


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, t, weight, loss_scale, **kargs):
        super().__init__()
        self.loss_function = {}
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        stu_encode, tea_encode = self(batch)
        losses = []
        for (loss_name, loss), scale in zip(self.loss_function.items(), self.hparams.loss_scale):
            if loss_name == 'kl':
                loss_res = loss(
                    F.softmax(stu_encode / self.hparams.t, dim=1).log(),
                    F.softmax(tea_encode / self.hparams.t, dim=1)
                ) * self.hparams.t ** 2

            elif loss_name == 'l1':
                loss_res = loss(stu_encode, tea_encode)
            else:
                loss_res = loss(stu_encode, tea_encode)
            loss_res *= scale
            self.log('loss/' + loss_name, loss_res.item(), on_step=True, on_epoch=True, prog_bar=False)
            losses.append(loss_res)

        if self.hparams.weight:
            assert len(self.hparams.weight) == len(
                losses), 'the number of self.weight should be the same as the number of loss'
            assert sum(self.hparams.weight) == 1, 'sum of wight should be 1, instead of {}'.format(
                sum(self.hparams.weight))
            total_loss = sum([loss * weight for loss, weight in zip(losses, self.hparams.weight)])
        else:
            total_loss = sum(losses) / len(losses)
        return total_loss

    def validation_epoch_end(self, outputs) -> None:
        add_similarity(self.model, self.logger.experiment, self.current_epoch, device=self.device)
        query = ['a man is riding bike', 'It\'s about seven o\'clock now', 'I want to find my computer',
                 'I want to see some cute cats!']
        random_query = np.random.randint(0, 4)
        imageQuery(query[random_query], self.model, self.encode_path, self.test_path, self.device, self.logger,
                   self.current_epoch)

    def validation_step(self, batch, batch_idx):
        img_tensor, captions, sentence = batch
        stu_encode, tea_encode = self(img_tensor)
        text_encode = self.model.teacher.encode_text(captions).float()
        losses = []
        for (loss_name, loss), scale in zip(self.loss_function.items(), self.hparams.loss_scale):
            if loss_name == 'kl':
                loss_res = loss(
                    F.softmax(stu_encode / self.hparams.t, dim=1).log(),
                    F.softmax(tea_encode / self.hparams.t, dim=1)
                ) * self.hparams.t ** 2
            elif loss_name == 'l1':
                loss_res = loss(stu_encode, tea_encode)
            else:
                stu_encode = stu_encode / stu_encode.norm(dim=1, keepdim=True)
                text_encode = text_encode / text_encode.norm(dim=1, keepdim=True)
                logits = (100 * stu_encode @ text_encode.T).softmax(dim=-1)
                label = torch.arange(len(logits), device=self.device)
                loss_res = loss(logits, label)
            loss_res *= scale
            self.log('val_loss/' + loss_name, loss_res.item(), on_step=True, on_epoch=True, prog_bar=False)
            losses.append(loss_res)

        if self.hparams.weight:
            assert len(self.hparams.weight) == len(
                losses), 'the number of self.weight should be the same as the number of loss'
            assert sum(self.hparams.weight) == 1, 'sum of wight should be 1, instead of {}'.format(
                sum(self.hparams.weight))
            total_loss = sum([loss * weight for loss, weight in zip(losses, self.hparams.weight)])
        else:
            total_loss = sum(losses) / len(losses)

        stu_encode = stu_encode / stu_encode.norm(dim=1, keepdim=True)
        text_encode = text_encode / text_encode.norm(dim=1, keepdim=True)

        stu_logits_per_image = (stu_encode @ text_encode.t())
        label = torch.arange(stu_logits_per_image.shape[0], device=self.device)

        for k in [1, 5, 10, 20, 30, 50]:
            acc = accuracy(stu_logits_per_image, label, top_k=k)
            self.log('val_acc/top{}'.format(k), acc, on_epoch=True, on_step=False, prog_bar=False)
        self.log('val_loss/' + 'total_loss', total_loss.item(), on_step=False, on_epoch=True, prog_bar=False)
        return

    def test_step(self, batch, batch_idx):

        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss
        if isinstance(loss, list):
            for l in loss:
                self.loss_function[l.lower()] = (self.get_loss_function(l.lower()))
        else:
            self.loss_function[loss.lower()] = self.get_loss_function(loss.lower())

    def get_loss_function(self, loss_name):
        if loss_name == 'l1':
            loss_function = nn.L1Loss()
        elif loss_name == 'ce':
            loss_function = nn.CrossEntropyLoss(reduction='mean')
        elif loss_name == 'kl':
            loss_function = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError("Invalid Loss Type!")
        return loss_function

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
