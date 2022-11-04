import torch
from torch.utils.data import Dataset
import imageio
import numpy as np
from kornia import create_meshgrid
from einops import rearrange
import cv2
import os
from typing import Tuple

import pickle


class ImageDataset(Dataset):
    def __init__(self, image_path: str, img_wh: Tuple[int, int]):
        img_wh = tuple(img_wh)
        image = cv2.imread(image_path, -1)
        image = image[:, :, None]
    
        self.coords = create_meshgrid(*image.shape[:2], True)[0] #[256, 256, 2]
        self.img = torch.FloatTensor(image) #[256, 256, 1]
        
        self.coords = rearrange(self.coords, 'h w c -> (h w) c')
        self.img = rearrange(self.img, 'h w c -> (h w) c')

    def __len__(self):
        return len(self.img)

    def __getitem__(self,idx):

        return {"coords": self.coords[idx], "img": self.img[idx]}
