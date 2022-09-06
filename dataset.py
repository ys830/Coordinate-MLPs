import torch
from torch.utils.data import Dataset
import imageio
import numpy as np
from kornia import create_meshgrid
from einops import rearrange
import cv2
from typing import Tuple


class ImageDataset(Dataset):
    def __init__(self, image_path: str, img_wh: Tuple[int, int], split: str):
        img_wh = tuple(img_wh)
        image = imageio.imread(image_path)/255.
        image = cv2.resize(image, img_wh)
        image = image[:, :, None]
    

        self.uv = create_meshgrid(*image.shape[:2], True)[0] #[112, 112, 2]
        self.rgb = torch.FloatTensor(image) #[112, 112, 1]

        if split == 'train':
            self.uv = self.uv[::2, ::2] #[56, 56, 2]
            self.rgb = self.rgb[::2, ::2] #[56, 56, 1]
        elif split == 'val':
            self.uv = self.uv[1::2, 1::2]
            self.rgb = self.rgb[1::2, 1::2]

        self.uv = rearrange(self.uv, 'h w c -> (h w) c')
        self.rgb = rearrange(self.rgb, 'h w c -> (h w) c')

        # image = imageio.imread(image_path)/255.
        
        # c = [image.shape[0]//2, image.shape[1]//2]
        # self.r = 56
        # image = image[c[0]-self.r:c[0]+self.r,
        #               c[1]-self.r:c[1]+self.r]
        # image = cv2.resize(image, (56, 56))
        

        # self.uv = create_meshgrid(self.r, self.r, True)[0] #[112, 112, 2]
        # self.rgb = torch.FloatTensor(image) #[112, 112, 1]

        # if split == 'train':
        #     self.uv = self.uv[::2, ::2] #[112, 112, 2]
        #     self.rgb = self.rgb[::2, ::2] #[112, 112, 1]
        
        # self.uv = rearrange(self.uv, 'h w c -> (h w) c')
        # self.rgb = rearrange(self.rgb, 'h w c -> (h w) c')


    def __len__(self):
        return len(self.uv)

    def __getitem__(self, idx: int):
        return {"uv": self.uv[idx], "rgb": self.rgb[idx]}