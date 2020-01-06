import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision import transforms, datasets

import os
import argparse
import shutil
from PIL import Image
import math
import time
import numpy as np

from model.SkrNet import *
from data.dataset import *
from utils.image import *
from utils.parse import *
from utils.utils import *


parser = argparse.ArgumentParser(description='SkrNet Object Detection training')
parser.add_argument('--model', type=str, default='SkrNet', metavar='model',
                    help='model to train (SkrNet,VGG16,ResNet18)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='batch size for each GPU during training (default: 16)')
parser.add_argument('--num-workers', default=32, type=int, metavar='N',
                    help='number of data loading threads (default: 32)')
parser.add_argument('--device', type=str, default='0', metavar='N',
                    help='device id')
parser.add_argument('--dataset', type=str, default='data/dji.data',
                    help='dataset (default: data/dji.data')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--optimizer', default='Adam', type=str, metavar='Optimizer',
                    help='optimizer: SGD, Adagrad, Adam, Adadelta, Adamax, ASGD, RMSprop')
parser.add_argument('--log', default='./logs/%s.log'%time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time())), type=str, metavar='PATH',
                    help='path to log (default: none)')
args = parser.parse_args()

def log(log_file,str):
    log = open(log_file,'a+')
    log.writelines(str+'\n') 
    log.close()

def train(model, data_loader, loss_func, optimizer):
    model.train()
    train_loss = 0.0
    total_batch = len(data_loader)
    ready_batch = 0 
    for img, target in data_loader:
        img, target = img.cuda(), target.cuda()
        img, target = Variable(img), Variable(target)
        optimizer.zero_grad()
        outputs = model(img)
        loss = loss_func(outputs, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        ready_batch += 1
        print("{}/{} ready/total".format(ready_batch, total_batch))
    train_loss /= float(len(data_loader))

    return train_loss

data_config = parse_data_config(args.dataset)
train_path  = data_config["train"]
valid_path  = data_config["valid"]

model = SkrNet()
num_gpu = len(args.device.split(','))
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
if(len(args.device)>1):
    model.to("cuda:{}".format(args.device.split(',')[0]))
    model = nn.DataParallel(model).cuda()
    region_loss = model.module.loss
else:
    model.to("cuda:{}".format(args.device))
    model.cuda()
    region_loss = model.loss



train_dataset = ListDataset(train_path)
valid_dataset = ListDataset(valid_path)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=num_gpu*args.batch_size, shuffle=True, num_workers=args.num_workers)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=int(num_gpu*args.batch_size/2), shuffle=True, num_workers=args.num_workers)

if(args.optimizer == 'SGD'):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
elif(args.optimizer == 'Adam'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(args.start_epoch, args.epochs):
    start = time.time()
    print('epoch%d...'%epoch)
    log(args.log,'epoch%d...'%epoch)
    log(args.log,str(optimizer))

    train_loss = train(model,train_loader,region_loss,optimizer)

    print('epoch%d time %.4fs\n' % (epoch,time.time()-start))
    log(args.log,'epoch%d time %.4fs\n' % (epoch,time.time()-start))