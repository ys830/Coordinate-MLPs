import torch
from torch import nn
from torch.utils.data import DataLoader
import os  
from collections import defaultdict
from tqdm import tqdm
import imageio
import argparse


from models import *

import metrics

from dataset import ImageDataset

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, default='images/fox.jpg',
                        help='path to the image to reconstruct')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[112, 112],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--arch', type=str, default='identity',
                        choices=['relu', 'ff', 'siren', 'gabor', 'bacon',
                                 'gaussian', 'quadratic', 'multi-quadratic',
                                 'laplacian', 'super-gaussian', 'expsin'],
                        help='network structure')
    parser.add_argument('--a', type=float, default=1.)
    parser.add_argument('--b', type=float, default=1.)
    parser.add_argument('--act_trainable', default=False, action='store_true',
                        help='whether to train activation hyperparameter')

    parser.add_argument('--sc', type=float, default=10.,
                        help='fourier feature scale factor (std of the gaussian)')
    parser.add_argument('--omega_0', type=float, default=30.,
                        help='omega in siren')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of epochs')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()

@torch.no_grad()
def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cuda'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_
    
def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)

if __name__ == "__main__":
    device = torch.device('cuda')
    args = get_opts()
    test_dataset = ImageDataset(args.image_path, args.img_wh)
    test_dataloader = DataLoader(test_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=args.img_wh[0]*args.img_wh[1],
                          pin_memory=False)

    model = Siren(first_omega_0=args.omega_0, hidden_omega_0=args.omega_0)
    load_ckpt(model, args.ckpt_path, model_name='mlp')
    model.cuda().eval()

    features_in_hook = []  # 勾的是指定层的输入
    features_out_hook = []  # 勾的是指定层的输出

    def hook(module, fea_in, fea_out):
        features_in_hook.append(fea_in)
        features_out_hook.append(fea_out)
   
    for (name, module) in model.named_modules():
        if "linear" in name:
            module.register_forward_hook(hook=hook)
                          
    for i, data in enumerate(test_dataloader):
        uv = data["uv"]
        rgb_gt = data["rgb"]
        uv, rgb_gt = uv.cuda(), rgb_gt.cuda()
        rgb_pre= model(uv)

    layer_output = dict()
    for i in range(len(features_out_hook)):
        layer_output[f'layer{i}'] = features_out_hook[i]
    
    print(layer_output)

