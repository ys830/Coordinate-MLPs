from unittest import TestLoader
import torch
from torch import nn
from einops import rearrange
from collections import OrderedDict
import torch.utils.hooks as hooks
from scipy.io import savemat
from tqdm import tqdm,trange


from opt import get_opts



# datasets
from dataset import ImageDataset
from torch.utils.data import DataLoader

# models
from models import PE, MLP, Siren, GaborNet, MultiscaleBACON

# metrics
from metrics import mse, psnr

# optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class CoordMLPSystem(LightningModule):
    def __init__(self, hparams, image_path, image_name):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.image_path = image_path
        self.image_name = image_name
        
        if hparams.use_pe:
            P = torch.cat([torch.eye(2)*2**i for i in range(10)], 1) # (2, 2*10)
            self.pe = PE(P)

        if hparams.arch in ['relu', 'gaussian', 'quadratic',
                            'multi-quadratic', 'laplacian',
                            'super-gaussian', 'expsin']:
            kwargs = {'a': hparams.a, 'b': hparams.b}
            act = hparams.arch
            if hparams.use_pe:
                n_in = self.pe.out_dim
            else:
                n_in = 2
            self.mlp = MLP(n_in=n_in, act=act,
                           act_trainable=hparams.act_trainable,
                           **kwargs)

        elif hparams.arch == 'ff':
            P = hparams.sc*torch.normal(torch.zeros(2, 256),
                                        torch.ones(2, 256)) # (2, 256)
            self.pe = PE(P)
            self.mlp = MLP(n_in=self.pe.out_dim)

        elif hparams.arch == 'siren':
            self.mlp = Siren(first_omega_0=hparams.omega_0,
                             hidden_omega_0=hparams.omega_0)

        elif hparams.arch == 'gabor':
            self.mlp = GaborNet(input_scale=max(hparams.img_wh)/4)

        elif hparams.arch == 'bacon':
            self.mlp = MultiscaleBACON(
                    frequency=[hparams.img_wh[0]//4, hparams.img_wh[1]//4])
        
    def forward(self, x):
        if hparams.use_pe or hparams.arch=='ff':
            x = self.pe(x)
        return self.mlp(x)

    def setup(self, stage=None):
        self.train_dataset = ImageDataset(self.image_path,
                                          hparams.img_wh)
                                          
        self.val_dataset = ImageDataset(self.image_path,
                                        hparams.img_wh)

        self.tset_dataset = ImageDataset(self.image_path,
                                        hparams.img_wh)
                                        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=False)
    
    def test_dataloader(self):
        return DataLoader(self.tset_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.img_wh[0]*self.hparams.img_wh[0],
                          pin_memory=False)

    def configure_optimizers(self):
        self.opt = Adam(self.mlp.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(self.opt, hparams.num_epochs, hparams.lr/1e2)

        return [self.opt], [scheduler]

    def training_step(self, batch, batch_idx):
        rgb_pred = self(batch['uv'])
        if hparams.arch=='bacon':
            loss = sum(mse(x, batch['rgb']) for x in rgb_pred)
            psnr_ = psnr(rgb_pred[-1], batch['rgb'])
        else:
            loss = mse(rgb_pred, batch['rgb'])
            psnr_ = psnr(rgb_pred, batch['rgb'])

        self.log('lr', self.opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        rgb_pred = self(batch['uv'])
        rgb_pred = torch.clamp(rgb_pred, min=0, max=1)
        
        if hparams.arch=='bacon':
            loss = mse(rgb_pred[-1], batch['rgb'], reduction='none')
        else:
            loss = mse(rgb_pred, batch['rgb'], reduction='none')

        log = {'val_loss': loss,
               'rgb_gt': batch['rgb']}

        if hparams.arch=='bacon':
            log['rgb_pred'] = rgb_pred[-1]
        else:
            log['rgb_pred'] = rgb_pred #（B，1）

        return log
    
    def test_step(self, batch, batch_idx):
        rgb_pred = self(batch['uv'])  
        mat_name = './mat_file/{}'.format(self.hparams.exp_name)
        if not os.path.exists(mat_name):
            os.makedirs(mat_name)
        mat_dir = os.path.join(mat_name, self.image_name + ".mat")
    
        layer_output = dict()
        for i in range(len(features_out_hook)):
            layer_output[f'layer{i}'] = features_out_hook[i].cpu().numpy()
        
        print(layer_output.keys())
        savemat(mat_dir,layer_output)

    def validation_epoch_end(self, outputs):
        mean_loss = torch.cat([x['val_loss'] for x in outputs]).mean()
        mean_psnr = -10*torch.log10(mean_loss)
        rgb_gt = torch.cat([x['rgb_gt'] for x in outputs])
        rgb_gt = rearrange(rgb_gt, '(h w) c -> c h w',
                           h=hparams.img_wh[1],
                           w=hparams.img_wh[0])
 
        rgb_pred = torch.cat([x['rgb_pred'] for x in outputs])
        rgb_pred = rearrange(rgb_pred, '(h w) c -> c h w',
                             h=hparams.img_wh[1],
                             w=hparams.img_wh[0])

        #记录图片
        self.logger.experiment.add_images('val/gt_pred',
                                          torch.stack([rgb_gt, rgb_pred]),
                                          self.global_step)
        #记录当前训练的步数：self.global_step  记录当前的epoch：self.current_epoch

        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/psnr', mean_psnr, prog_bar=True)

if __name__ == '__main__':
    hparams = get_opts()

    for filepath,dirnames,filenames in os.walk(hparams.image_file):
        for filename in tqdm(filenames, desc="image_number"):
            features_in_hook = []  # 勾的是指定层的输入
            features_out_hook = []  # 勾的是指定层的输出

            image_path = os.path.join(filepath,filename)
            image_name = filename.split(".")[0]
            system = CoordMLPSystem(hparams, image_path, image_name)

            pbar = TQDMProgressBar(refresh_rate=1)
            callbacks = [pbar]

            logger = TensorBoardLogger(save_dir="new_logs",
                                        name=hparams.exp_name,
                                        version= image_name, 
                                        default_hp_metric=False)

            trainer = Trainer(max_epochs=hparams.num_epochs,
                                callbacks=callbacks,
                                logger=logger,
                                enable_model_summary=True,
                                accelerator='auto',
                                devices=1,
                                num_sanity_val_steps=0,
                                log_every_n_steps=1,
                                check_val_every_n_epoch=20,
                                benchmark=True)

            trainer.fit(system)
            for (name, module) in system.named_modules():
                if "linear" in name:
                    module.register_forward_hook(hook=hook)
            trainer.test(system)
          

