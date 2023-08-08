import torch
from torch import nn,optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import argparse
import json
import utilits

parser = argparse.ArgumentParser(description='Prediction process')
parser.add_argument('input_img', default='./flowers/test/10/image_07090.jpg', nargs=1, action="store", type = str)
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs=1, action="store",type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

argParser = parser.parse_args()
path_img = argParser.input_img
no_of_outputs = argParser.top_k
mode = argParser.gpu
path = argParser.checkpoint
json_file = argParser.category_names


traindata,trainloader,validloader,testloader = utilits.load_data(["./flowers"])
model = utilits.load_checkpoint(path)
with open(json_file, 'r') as json_file:
    cat_to_name = json.load(json_file)
    
probs,classes= utilits.predict(path_img,model,no_of_outputs,mode)
classes = [cat_to_name[c] for c in classes]

for i in range(no_of_outputs):
    print(" {}. {} : {:.3f}".format(i+1,classes[i],probs[i]))
    