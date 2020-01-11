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
parser.add_argument('--batch', type=int, default=32, metavar='N',
                    help='batch size for each GPU during training (default: 32)')
parser.add_argument('--workers', default=32, type=int, metavar='N',
                    help='number of data loading threads (default: 32)')
parser.add_argument('--device', type=str, default='0', metavar='N',
                    help='device id')
parser.add_argument('--dataset', type=str, default='data/dji.data',
                    help='dataset (default: data/dji.data')
parser.add_argument('--end', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                    help='path to pretrain checkpoint (default: none)')
parser.add_argument('--optimizer', default='Adam', type=str, metavar='Optimizer',
                    help='optimizer: SGD, Adagrad, Adam, Adadelta, Adamax, ASGD, RMSprop')
parser.add_argument('--log', default='./logs/%s.log'%time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time())), type=str, metavar='PATH',
                    help='path to log (default: none)')
args = parser.parse_args()

def log(log_file,str):
    log = open(log_file,'a+')
    log.writelines(str+'\n') 
    log.close()

def test(model, data_loader):
    model.eval()
    avg_iou = 0.0
    for img, target in data_loader:
        img, target = img.cuda(), target.cuda()
        img, target = Variable(img), Variable(target)
        output = model(img)
        avg_iou += evaluate(output, target)
    avg_iou /= float(len(data_loader))
    return avg_iou


def train(model, data_loader, loss_func, optimizer):
    model.train()
    avg_loss, avg_recall50, avg_recall75, avg_iou = 0.0, 0.0, 0.0, 0.0
    total_batch = len(data_loader)
    ready_batch = 0 
    for img, target in data_loader:
        img, target = img.cuda(), target.cuda()
        img, target = Variable(img), Variable(target)
        optimizer.zero_grad()
        outputs = model(img)
        loss, recall50, recall75, iou = loss_func(outputs, target)
        avg_loss += loss.item()
        avg_recall50 += recall50
        avg_recall75 += recall75
        avg_iou += iou
        loss.backward()
        optimizer.step()
        ready_batch += 1
        print("{}/{} ready/total".format(ready_batch, total_batch))
    avg_loss /= float(len(data_loader))
    avg_recall50 /= float(len(data_loader))
    avg_recall75 /= float(len(data_loader))
    avg_iou /= float(len(data_loader))
    return avg_loss, avg_recall50, avg_recall75, avg_iou

data_config = parse_data_config(args.dataset)
train_path  = data_config["train"]
valid_path  = data_config["valid"]

if(args.pretrain):
    model = SkrNet(detection = False)
    model.load_state_dict(torch.load(args.pretrain))
    model.detection = True
    print('load pretrain model')
else:
    model = SkrNet()

num_gpu = len(args.device.split(','))
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
if(len(args.device)>1):
    model.to("cuda:{}".format(args.device.split(',')[0]))
    device_ids = [int(device) for device in args.device.split(',')]
    model = nn.DataParallel(model,device_ids=device_ids).cuda()
    region_loss = model.module.loss
    # region_loss = nn.DataParallel(model.module.loss,device_ids=device_ids).cuda()
else:
    model.to("cuda:{}".format(args.device))
    model.cuda()
    region_loss = model.loss

train_dataset = ListDataset(train_path)
valid_dataset = ListDataset(valid_path)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=num_gpu*args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=int(num_gpu*args.batch/2), shuffle=True, num_workers=args.workers, pin_memory=True)

if(args.optimizer == 'SGD'):
    optimizer = torch.optim.SGD(model.parameters())
elif(args.optimizer == 'Adam'):
    optimizer = torch.optim.Adam(model.parameters())

history_score = np.zeros((args.end + 1,4))
for epoch in range(args.start, args.end):
    start = time.time()
    print('epoch%d...'%epoch)
    log(args.log,'epoch%d...'%epoch)
    log(args.log,str(optimizer))

    loss, recall50, recall75, avg_iou = train(model,train_loader,region_loss,optimizer)
    print('training: avg loss: %f, avg recall50: %f, avg recall75:%f, avg iou:%f\n' % (loss,recall50,recall75,avg_iou))
    log(args.log,'training: avg loss: %f, avg recall50: %f, avg recall75:%f, avg iou:%f\n' % (loss,recall50,recall75,avg_iou))
    iou = test(model, train_loader)
    print('testing: avg iou: %f\n' % iou)
    log(args.log,'testing: avg iou: %f\n' % iou)
    if iou > max(history_score[:,3]):
        torch.save(model.module.state_dict(), './checkpoint/detection/%s_%.4f.pkl'%(args.model,iou))
    history_score[epoch][0] = loss
    history_score[epoch][1] = recall50
    history_score[epoch][2] = recall75
    history_score[epoch][3] = iou
    print('epoch%d time %.4fs\n' % (epoch,time.time()-start))