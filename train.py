from unittest import TestLoader
import torch
from torch import nn
from einops import rearrange
from collections import OrderedDict
import torch.utils.hooks as hooks
from tqdm import tqdm,trange


from opt import get_opts



# datasets
from dataset import ImageDataset
from torch.utils.data import DataLoader

# models
from models import PE, MLP, Siren, GaborNet, MultiscaleBACON, FP

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


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class CoordMLPSystem(LightningModule):
    def __init__(self, hparams, image_path, image_name):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.image_path = image_path
        self.image_name = image_name

        self.FP_train = FP(hparams.num_projections, 1, train= True)
        self.FP_test = FP(hparams.test_num_projections, 1, train= False)
        
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
                          shuffle=False,
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
        # img_pred = self(batch['coords']) #[(h w),2] ->[(h w), 1]
        # img_pred = rearrange(img_pred, '(h w) c -> c h w', h =256)[None, ...] #[(h w), 1] -> [1,1,h,w]
        # sino_pred = self.FP_train(img_pred).squeeze(0) #[1,1,h,w] -> [h,w]
        # batch['img'] = rearrange(batch['img'], '(h w) c -> c h w', h =256)[None, ...]
        # sino_gt = self.FP_train(batch['img']).squeeze(0)
        
        coords, img = batch["coords"], batch["img"] #[b, 256, 256, 2] #是否需要归一化-1~1？
        img = rearrange(batch["img"], '(h w) c -> c h w', h =256)[None, ...]
        coords = rearrange(batch["coords"], '(h w) c -> h w c', h =256)[None, ...]

        pre = self(coords)
        pre = pre.unsqueeze(1) #[1,1,256,256]
        pre_proj = self.FP_train(pre) #[b,h,w]
        pre_proj = pre_proj.squeeze(0)

        img_proj = self.FP_train(img)
        img_proj = img_proj.squeeze(0)

        loss = mse(pre_proj, img_proj)
        psnr_ = psnr(pre_proj, img_proj)

        self.log('lr', self.opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # img_pred = self(batch['coords'])
        # img_pred_reshape = rearrange(img_pred, '(h w) c -> c h w', h =256)[None, ...]
        # sino_pred = self.FP_test(img_pred_reshape).squeeze(0)
        # img = rearrange(batch['img'], '(h w) c -> c h w', h =256)[None, ...]
        # sino_gt = self.FP_test(img).squeeze(0)
        
        coords, img = batch["coords"], batch["img"] #[b, 256, 256, 2] #是否需要归一化-1~1？
        img = rearrange(batch["img"], '(h w) c -> c h w', h =256)[None, ...]
        coords = rearrange(batch["coords"], '(h w) c -> h w c', h =256)[None, ...]
        
        test_pre = self(coords) 
        test_pre = test_pre.unsqueeze(1) #[1,1,256,256]
        test_pre_proj = self.FP_test(test_pre)
        # test_pre_proj = test_pre_proj.squeeze(0)
        test_img_proj = self.FP_test(img)
        # test_img_proj = test_img_proj.squeeze(0)
        
        loss = mse(test_img_proj, test_pre_proj, reduction='none')

        log = {'val_loss': loss,
               'rgb_gt': img.squeeze(0)*255}

        log['img_pred'] = test_pre.squeeze(0)*255

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
        # rgb_gt = torch.cat([x['rgb_gt'] for x in outputs])
        # rgb_gt = rearrange(rgb_gt, '(h w) c -> c h w',
        #                    h=hparams.img_wh[1],
        #                    w=hparams.img_wh[0])

        rgb_gt = outputs[0]['rgb_gt']
        rgb_pred = outputs[0]['img_pred']

 
        # rgb_pred = torch.cat([x['img_pred'] for x in outputs])
        # rgb_pred = rearrange(rgb_pred, '(h w) c -> c h w',
        #                      h=hparams.img_wh[1],
        #                      w=hparams.img_wh[0])

        #记录图片
        self.logger.experiment.add_images('val/gt_pred',
                                          torch.stack([rgb_gt, rgb_pred]),
                                          self.global_step)
        #记录当前训练的步数：self.global_step  记录当前的epoch：self.current_epoch

        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/psnr', mean_psnr, prog_bar=True)

if __name__ == '__main__':
    hparams = get_opts()

    image_name = 'img0'
    system = CoordMLPSystem(hparams, hparams.image_path, image_name)

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
          

