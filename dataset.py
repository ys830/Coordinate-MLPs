import torch
from torch.utils.data import Dataset
import imageio
import numpy as np
from kornia import create_meshgrid
from einops import rearrange
import cv2
import os
from typing import Tuple



class ImageDataset(Dataset):
    def __init__(self, image_path: str, img_wh: Tuple[int, int]):
        img_wh = tuple(img_wh)
        image = imageio.imread(image_path)/255.
        image = image[:, :, None]
    
        self.uv = create_meshgrid(*image.shape[:2], True)[0] #[112, 112, 2]
        self.rgb = torch.FloatTensor(image) #[112, 112, 1]

        self.uv = rearrange(self.uv, 'h w c -> (h w) c')
        self.rgb = rearrange(self.rgb, 'h w c -> (h w) c')


    def __len__(self):
        return len(self.uv)

    def __getitem__(self, idx: int):
        return {"uv": self.uv[idx], "rgb": self.rgb[idx]}