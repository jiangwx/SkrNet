
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

def reshape(img, shape, bbox):

    w, h = img.size
    if((h/w)>(shape[0]/shape[1])):
        _w = int(w*shape[0]/h)
        _h = shape[0]
    else:
        _w = shape[1]
        _h = int(h*shape[1]/w)
    pad_h, pad_w = int((shape[0] - _h)/2), int((shape[1] - _w)/2)

    img_resize = img.resize((_w,_h))
    img_tensor = transforms.ToTensor()(img_resize)

    pad = (pad_w, pad_w, pad_h, pad_h)
    img_pad = F.pad(img_tensor, pad, "constant", value = 0)

    bbox[0] = int(bbox[0]*(_h/h)) + pad_w
    bbox[1] = int(bbox[1]*(_h/h)) + pad_h
    bbox[2] = int(bbox[2]*(_h/h)) + pad_w
    bbox[3] = int(bbox[3]*(_h/h)) + pad_h

    return img_pad, bbox

class ListDataset(Dataset):
    def __init__(self, img_list, multiscale=True):
        
        self.img_list = open(img_list,'r').readlines
        self.multiscale = multiscale

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        img_path = self.img_list[index % len(self.img_list)].rstrip().split(' ')[0]
        labels = self.img_list[index % len(self.img_list)].rstrip().split(',')[1].strip('[').strip(']').split(' ')
        bbox = [float(label) for label in labels]

        img = Image.open(img_path).convert('RGB')
        
        if(self.multiscale):
            hight = (random.randint(-3,3)+20)*8
            width = (random.randint(-3,3)+20)*16
        else:
            hight = 160
            width = 320
        self.shape = (hight, width)

        img, bbox = reshape(img, self.shape, bbox)

        return img, bbox

# TO DO: data augmentation
