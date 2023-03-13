# cse455-final-project
This is the code for CSE455 Final Project, "Birds Birds Birds - Are they real?".
The details of this project can be found [here](https://mfj229.github.io/cse455-final-project/).

## Preparation
```Python
import os
import shutil

original_folder ="./birds/train"
train_folder = "./preprocessed_data/train"
val_folder = "./preprocessed_data/val"

def copy(category):
    print ("copying ", category)

    origial_path = os.path.join(original_folder, str(category))
    trian_path = os.path.join(train_folder, str(category))
    val_path = os.path.join(val_folder, str(category))

    if not os.path.exists(trian_path):
      os.mkdir(trian_path)
    if not os.path.exists(val_path):
      os.mkdir(val_path)

    # get category file names
    files = os.listdir(origial_path)

    # get count of train and val files for splitting train and validation image files
    train_file_count = int(len(files) * 0.8)
    val_file_count = len(files) - train_file_count
    
    # copy train files
    for i in range(train_file_count):
        src_path = os.path.join(origial_path, files[i])
        dst_path = os.path.join(trian_path, files[i])
        shutil.copy(src_path, dst_path)

    # copy val files
    for i in range(val_file_count):
        src_path = os.path.join(origial_path, files[train_file_count+i])
        dst_path = os.path.join(val_path, files[train_file_count+i])
        shutil.copy(src_path, dst_path)

if not os.path.exists("./preprocessed_data"):
  os.mkdir("./preprocessed_data")
if not os.path.exists("./preprocessed_data/train"):
  os.mkdir("./preprocessed_data/train")
if not os.path.exists("./preprocessed_data/val"):
  os.mkdir("./preprocessed_data/val")

# 0-554
for i in range(555): 
    copy(i)
```

## Training
```Python
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import datetime

def train_and_validate(model, loss_criterion, optimizer, scheduler, epochs=25):
    '''
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    ''' 
    history = []
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()  
        train_loss = 0.0
        train_correct = 0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update statistics
            train_loss += loss.item() * inputs.size(0) # batch size!!!
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100*train_correct / len(train_dataset) #  train accuracy per epoc
        train_loss = train_loss / len(train_dataset) # average traing loss per epoc
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            valid_correct = 0
            for j, (inputs, labels) in enumerate(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)
                 # Update statistics
                valid_loss += loss.item() * inputs.size(0) # batch size!!!
                _, predicted = torch.max(outputs.data, 1)
                valid_correct += (predicted == labels).sum().item()
            
            valid_acc = 100*valid_correct / len(val_dataset) #  val accuracy per epoc
            valid_loss = valid_loss / len(val_dataset) # average va loss per epoc

        history.append([train_loss, train_acc, valid_loss, valid_acc])
                
        # Save if the model has best accuracy till now
        checkpoint_filename = 'model_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + str(epoch)+ '(' + str(train_acc)[0:5] + '_' + str(valid_acc)[0:5] + ')' + '.pt'
        torch.save(model.state_dict(), checkpoint_filename) #model.load_state_dict(torch.load(filepath), strict=False)

        last_lr = ""
        if scheduler is not None:
          last_lr = scheduler.get_last_lr()
          scheduler.step()
  
        print("Epoch: {}/{} time: {:.4f}s".format(epoch, epochs, time.time() - epoch_start))
        print("\ttrain_loss {:.4f} train_acc {:.4f}% val_loss {:.4f} val_acc {:.4f}% lr {}".format(train_loss, train_acc, valid_loss, valid_acc, last_lr))
        print("\t", checkpoint_filename)



    return model, history

# Define the transform for the training dataset
torch.manual_seed(1)
mean = [0.485, 0.456, 0.406] # pretrained
std= [0.229, 0.224, 0.225] # pretrained
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=384, scale=(0.70, 1.0)), 
        #transforms.RandomRotation(degrees=15),
        #transforms.Resize(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.CenterCrop(size=384),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=384),
        transforms.CenterCrop(size=384),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=384),
        transforms.CenterCrop(size=384),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

root = "./preprocessed_data"
test_image_root="./birds/test"

# Create the training&validation dataset with the appropriate transforms
train_dataset = torchvision.datasets.ImageFolder(root=root+'/train', transform=image_transforms["train"])
val_dataset = torchvision.datasets.ImageFolder(root=root+'/val', transform=image_transforms["val"])
test_dataset = torchvision.datasets.ImageFolder(root=test_image_root, transform=image_transforms["test"])

# Create the data loaders for the training and validation datasets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

classidx_to_birdidx = {v: k for k, v in train_dataset.class_to_idx.items()}

num_classes = 555

# training start
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)

torch.manual_seed(1)

#model = models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
model = models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
print(model)
# Make all layer parameters untrainable
#for param in model.parameters():
#    param.requires_grad = False
model.classifier[0] = nn.Dropout(0.5)
model.classifier[1] = nn.Linear(1280, num_classes)

# Convert model to be used on GPU
model = model.to(device)

# Define Optimizer and Loss Function
loss_func = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.002)
scheduler = StepLR(optimizer,  step_size = 2,   gamma = 0.7) 

# Train the model for 40 epochs
num_epochs = 40
trained_model, history = train_and_validate(model, loss_func, optimizer, scheduler, num_epochs)

# weight_decay = 0.00002, AdamW
```

## Prediction
```Python
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math
import datetime
from PIL import Image
import csv

def read_filelines(filename):
  with open(filename, 'r') as file:
    lines = [line.rstrip('\n') for line in file.readlines()]
  return lines

def read_csv_file(filename):
    with open(filename, newline='\n') as file:
        reader = csv.reader(file)
        lines = list(reader)
    return lines

def write_csv_file(lines, output_file):
    with open(output_file, 'w', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerows(lines)

def predict(device, model, filename, classidx_to_birdidx, idx_to_name, transform):
    # Load the image and apply the transformation
    image = Image.open(filename)
    image = transform(image)
    
    # Add a batch dimension to the tensor
    image = image.unsqueeze(0)
    image = image.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    # Disable autograd to save memory and improve performance
    with torch.no_grad():
        # Forward pass through the model
        output = model(image)
        
        # Get the predicted class index
        _, predicted = torch.max(output.data, 1)
        classidx = predicted[0].item()
        birdidx = int(classidx_to_birdidx[classidx])
        name = idx_to_name[birdidx]

        return birdidx, name


# Define the transform for the training dataset
mean = [0.485, 0.456, 0.406] # pretrained
std= [0.229, 0.224, 0.225] # pretrained
image_transforms = { 
    'train': transforms.Compose([
        transforms.Resize(size=384),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=384),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=384),
        transforms.CenterCrop(size=384),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=384),
        transforms.CenterCrop(size=384),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

root = "./preprocessed_data"
test_image_root="./birds/test"

# Create the training&validation dataset with the appropriate transforms
train_dataset = torchvision.datasets.ImageFolder(root=root+'/train', transform=image_transforms["train"])
val_dataset = torchvision.datasets.ImageFolder(root=root+'/val', transform=image_transforms["val"])
test_dataset = torchvision.datasets.ImageFolder(root=test_image_root, transform=image_transforms["test"])

# Create the data loaders for the training and validation datasets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

classidx_to_birdidx = {v: k for k, v in train_dataset.class_to_idx.items()}

num_classes = 555

idx_to_name = read_filelines("./birds/names.txt")

# preparing model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# get model
#model_20230309_193650_17(98.55_87.68).pt
model_filename = "model_20230310_121058_21_98.91_88.80" # "model_20230309_040120_10_96.02_86.34"

#model = models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
#model.heads[0] = nn.Linear(1024, num_classes)
model = models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
model.classifier[0] = nn.Dropout(0.5)
model.classifier[1] = nn.Linear(1280, num_classes)

model.load_state_dict(torch.load(model_filename+".pt"), strict=False)
model = model.to(device)

test_image_list = read_csv_file("./birds/sample.csv")

for image_row in test_image_list:
  filename = image_row[0]
  birdidx = image_row[1]

  if birdidx != 'class' :
     image_path =  './birds/test/0/' +  filename[5:]
     birdidx, _ = predict( device, model, image_path, classidx_to_birdidx, idx_to_name, image_transforms["test"] )
     image_row[1] = str(birdidx)

write_csv_file(test_image_list, "./" + model_filename + "_result_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv")
```
