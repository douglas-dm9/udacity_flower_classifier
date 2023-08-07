#Import Required Libraries
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import argparse
import utilits

#Creating Arguments for CLI
parser = argparse.ArgumentParser(description='Neural Network Training')
parser.add_argument('data_dir', nargs=1, action="store", default=["./flowers"])
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", type=float, action="store", default=0.001)
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
parser.add_argument('--arch', dest="arch", action="store"   , default="vgg16", type=str)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)

pa = parser.parse_args()

    
if(pa.epochs<=0):
    print("Number of ephocs not valid, should be greater than 0")

where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structre = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
process_unit = pa.gpu
epochs = pa.epochs

train_data,trainloader,validloader,testloader = utilits.load_data(where)
model,criterion,optimizer = utilits.build_network(structre,lr,dropout,hidden_layer1,process_unit)
utilits.train_network(model, criterion, optimizer,trainloader, validloader, epochs, 25, process_unit)
utilits.save_checkpoint(model,train_data,path,structre,hidden_layer1,dropout,lr,epochs)

print("Training and saving model succesfully!!")