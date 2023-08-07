import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import argparse
from PIL import Image

# Function to return dataloaders 
def load_data(where):
    #Folders
    for dir in where:
        where = str(dir)
    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Transforms
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    #Load the datasets
    #The datasets are the same as used on the Part-8 Transfer Learning Notebook
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    #D   trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=50)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=50)
    
    return train_data, trainloader, validloader, testloader

#Build Network function
def build_network(arch,lr=0.001,dropout=0.5,hidden_units=4096,mode='gpu'):
    
    if(arch=='vgg16'):
        model = models.vgg16(pretrained=True)
    elif(arch=='densenet121'):
        model = models.densenet121(pretrained=True)
    else:
        print("Choose between vgg16 and densenet121")
    
    for param in model.parameters():
        param.requires_grad = False
        if(arch=="vgg16"):
            model.classifier = nn.Sequential(OrderedDict([
                                     ('fc1',nn.Linear(25088,hidden_units,bias=True)),
                                     ('relu1',nn.ReLU()),
                                     ('drop1',nn.Dropout(p=0.5)),
                                     ('fc2',nn.Linear(hidden_units,102,bias=True)),
                                     ('softmax1',nn.LogSoftmax(dim=1))]))
        elif(arch=="densenet121"):
            model.classifier = nn.Sequential(OrderedDict([
                                     ('fc1',nn.Linear(1024,hidden_units,bias=True)),
                                     ('relu1',nn.ReLU()),
                                     ('drop1',nn.Dropout(p=0.5)),
                                     ('fc2',nn.Linear(hidden_units,102,bias=True)),
                                     ('softmax1',nn.LogSoftmax(dim=1))]))
        else:
            print("Choose between vgg16 or densenet121 !")
            
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        
        if torch.cuda.is_available() and mode == 'gpu':
            model.cuda()
            
        return model, criterion, optimizer

#Validate Model
def validate_model(model, criterion, dataloader,mode='gpu'):
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for images,labels in iter(dataloader):
            if torch.cuda.is_available() and mode=='gpu':
                device = 'cuda'
                images,labels = images.to(device),labels.to(device)
            
            output = model.forward(images)
            loss += criterion(output, labels).item()

            ps = torch.exp(output) 
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
               
    return loss, accuracy

#Training Network
def train_network(model,criterion,optimizer,trainloader, validloader, epochs=25,print_every=25,mode='gpu'):
    batch=0
    running_loss=0
    
    print("----------------Training Started------------------\n")

    for e in range(epochs):
        for images, labels in  trainloader:
            batch+=1
            if torch.cuda.is_available() and mode=='gpu':
                device='cuda'
                images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            #Forward and Backward Propogation
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch % print_every == 0:
                validation_loss = 0
                validation_accuracy = 0
                model.eval()

                #VALIDATION
                with torch.no_grad():
                    validation_loss, validation_accuracy = validate_model(model, criterion, validloader)

                print(f"Epoch {e+1}/{epochs}.. "
                      f"Training loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {validation_accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()

    print("\n---------------Training Completed---------------")
    
    
#Save model checkpoint
def save_checkpoint(model,train_data ,path='./checkpoint.pth',structure ='vgg16', hidden_layer1=4096,dropout=0.5,lr=0.001,epochs=25):
    
    model.cpu
    model.class_to_idx = train_data.class_to_idx
    chpt = {'structure' :structure,
            'hidden_layer1':hidden_layer1,
            'dropout':dropout,
            'lr':lr,
            'no_of_epochs':epochs,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx}
    
    chpt['classifier'] = model.classifier
    if(path!="./checkpoint.pth"):
        path = path + "/checkpoint.pth"
    torch.save(chpt,path)
    print("Model Saved")

#Load model checkpoint
def load_checkpoint(path='checkpoint.pth'):
    for i in path:
        path=str(i)
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']
    model,_,_ = build_network(structure,lr,dropout,hidden_layer1)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

#Pre Process PIL Image
def process_image(img_path):
    for i in img_path:
        path = str(i)
    img = Image.open(path)
    process = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])
    final_img = process(img)
    return final_img

# Predict image class
def predict(image_path, model, topk=5,process_unit='gpu'):
    #if torch.cuda.is_available() and process_unit=='gpu':
      #model.to('cuda:0')
    model.eval()
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    if process_unit == 'gpu':
        with torch.no_grad():
            output = model.forward(img.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img)
    probability = torch.exp(output)
    return probability.topk(topk)



