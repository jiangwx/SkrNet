import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
from torch.utils.data import DataLoader,Dataset

import os
import argparse
import shutil
from PIL import Image
import math
import time
import numpy as np

from model.SkrNet import *


parser = argparse.ArgumentParser(description='SkrNet Pretraining')
parser.add_argument('--model', type=str, default='SkrNet', metavar='model',
                    help='model to train (SkyNet,VGG16,ResNet18)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='batch size for each GPU during training (default: 16)')
parser.add_argument('--num-workers', default=32, type=int, metavar='N',
                    help='number of data loading threads (default: 32)')
parser.add_argument('--device', type=str, default='0,1,2,3,4,5,6,7', metavar='N',
                    help='device id')                  
parser.add_argument('--dataset', type=str, default='/media/DATASET/mini-imagenet',
                    help='training dataset (default: /media/DATASET/mini-imagenet')
parser.add_argument('--end', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start', default=0, type=int, metavar='N',
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

def decay(base_lr,epoch):
    lr = base_lr *  (0.1 ** (epoch // 25))
    return lr

def poly(base_lr, power, total_epoch, now_epoch):
    return base_lr * (1 - math.pow(float(now_epoch) / float(total_epoch), power))

def test(model, data_loader, loss_func):
    model.eval()
    correct = 0
    loss = 0.0

    for inputs, labels in data_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        loss += loss_func(outputs, labels).item()

    accuracy = 100. * float(correct) / float(len(data_loader.dataset))
    loss /= float(len(data_loader))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(loss, correct, len(data_loader.dataset), accuracy))
    log(args.log,'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, loss

def train(model, data_loader, loss_func, optimizer):
    model.train()
    correct = 0
    train_loss = 0.0
    total_batch = len(data_loader)
    ready_batch = 0 
    for inputs, labels in data_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        loss.backward()
        optimizer.step()
        ready_batch += 1
        print("{}/{} ready/total".format(ready_batch, total_batch))

    accuracy = 100. * float(correct) / float(len(data_loader.dataset))
    train_loss /= float(len(data_loader))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(train_loss, correct, len(data_loader.dataset), accuracy))
    log(args.log,'Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(train_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, train_loss

print(args)
log(args.log,str(args))

train_data_transform = transforms.Compose([
    transforms.RandomResizedCrop(320),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_data_transform = transforms.Compose([
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(train_data_transform,test_data_transform)

log(args.log,str(train_data_transform))
log(args.log,str(test_data_transform))


model = SkrNet(detection = False)
print(model)

log(args.log,str(model))
if args.start != 0:
    model.load_state_dict(torch.load(args.resume))
    

num_gpu = len(args.device.split(','))
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
if(len(args.device)>1):
    model.to("cuda:{}".format(args.device.split(',')[0]))
    model = nn.DataParallel(model).cuda()
else:
    model.cuda()

train_dataset = torchvision.datasets.ImageFolder(root=args.dataset+'/train', transform=train_data_transform)
test_dataset = torchvision.datasets.ImageFolder(root=args.dataset+'/val', transform=test_data_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=num_gpu*args.batch_size, shuffle=True, num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(num_gpu*args.batch_size/2), shuffle=True, num_workers=args.num_workers)

history_score=np.zeros((args.end + 1,4))

loss_func = nn.CrossEntropyLoss()

if(args.optimizer == 'SGD'):
    optimizer = torch.optim.SGD(model.parameters())
elif(args.optimizer == 'Adam'):
    optimizer = torch.optim.Adam(model.parameters())
elif(args.optimizer == 'RMSprop'):
    optimizer = torch.optim.RMSprop(model.parameters())
elif(args.optimizer == 'Adagrad'):
    optimizer = torch.optim.Adagrad(model.parameters())
elif(args.optimizer == 'Adadelta'):
    optimizer = torch.optim.Adadelta(model.parameters())
elif(args.optimizer == 'Adamax'):
    optimizer = torch.optim.Adamax(model.parameters())
elif(args.optimizer == 'AdamW'):
    optimizer = torch.optim.AdamW(model.parameters())
elif(args.optimizer == 'RAdam'):
    optimizer = RAdam(model.parameters())
elif(args.optimizer == 'SparseAdam'):
    optimizer = torch.optim.SparseAdam(model.parameters())
elif(args.optimizer == 'Ranger'):
    optimizer = Ranger(model.parameters())
elif(args.optimizer == 'ASGD'):
    optimizer = torch.optim.ASGD(model.parameters())
else:
    print('please specify a valid optimizer')

for epoch in range(args.start, args.end):
    start = time.time()
    print('epoch%d...'%epoch)
    log(args.log,'epoch%d...'%epoch)
    log(args.log,str(optimizer))

    train_accuracy, train_loss = train(model,train_loader,loss_func,optimizer)
    test_accuracy, test_loss = test(model, test_loader, loss_func)

    if test_accuracy > max(history_score[:,2]):
        torch.save(model.module.state_dict(), './checkpoint/pretrain/%s_%.4f.pkl'%(args.model,test_accuracy))
    history_score[epoch][0] = train_accuracy
    history_score[epoch][1] = train_loss
    history_score[epoch][2] = test_accuracy
    history_score[epoch][3] = test_loss

    print('epoch%d time %.4fs\n' % (epoch,time.time()-start))
    log(args.log,'epoch%d time %.4fs\n' % (epoch,time.time()-start))