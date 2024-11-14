# Importing torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F#All activation functions are present here
import torch.optim as optim # Optimizer is stored here
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
import torchvision.models as models
from torchviz import make_dot  #visualize computational graph
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Importing other packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import os
from PIL import Image

import torchvision.models as models
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# To test whether GPU instance is present in the system or not.
use_cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)

device = torch.device("cuda" if use_cuda else "cpu")
print(device)

import os

# Replace these paths with your local directory paths where you extracted the zip file
dataset = {
    "train_data": "/home/xintrean/Back-up/notes/sem_5/Project_1/Leaf/leaf/leaf/train",
    "valid_data": "/home/xintrean/Back-up/notes/sem_5/Project_1/Leaf/leaf/leaf/test", 
    #"test_data": "/home/xintrean/Back-up/notes/sem_5/Project_1/Trunk/trunk/test"
}

all_data = []
for path in dataset.values():
    data = {"imgpath": [] , "labels": [] }
    category = os.listdir(path)

    for folder in category:
        folderpath = os.path.join(path , folder)
        filelist = os.listdir(folderpath)
        for file in filelist:
            fpath = os.path.join(folderpath, file)
            data["imgpath"].append(fpath)
            data["labels"].append(folder)


    all_data.append(data.copy())
    data.clear()



train_df = pd.DataFrame(all_data[0] , index=range(len(all_data[0]['imgpath'])))
valid_df = pd.DataFrame(all_data[1] , index=range(len(all_data[1]['imgpath'])))


#Convert labels to numbers
lb = LabelEncoder()
train_df['encoded_labels'] = lb.fit_transform(train_df['labels'])
valid_df['encoded_labels'] = lb.fit_transform(valid_df['labels'])

valid_df , test_df = train_test_split(valid_df ,  train_size= 0.95 , shuffle=True, random_state=124)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Size of train df {len(train_df)}")
print(f"Size of valid df {len(valid_df)}")
print(f"Size of test df {len(test_df)}")


#batch_size = 32
#image_size = (224, 224)
BATCH_SIZE = 8
IMAGE_SIZE = (224, 224)

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'imgpath']
        image = Image.open(img_path).convert('RGB')
        label = self.df.loc[idx, 'encoded_labels']
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),               # Random horizontal flip
    transforms.RandomRotation(10),                   # Random rotation up to 10 degrees
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Random zoom
    transforms.ColorJitter(contrast=0.1),            # Random contrast adjustment
    transforms.ToTensor(),                           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(train_df, transform=train_transform)
valid_dataset = CustomDataset(valid_df, transform=transform)
test_dataset = CustomDataset(test_df, transform=transform)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(valid_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for images, labels in train_loader:
    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)
    #print("Labels batch values:", labels)
    #print("Images[0]", images[0])
    #print("Labels[0]", labels[0])
    break  # Just print for one batch

for images, labels in valid_loader:
    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)
    break


for images, labels in test_loader:
    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)
    break


# model = model = models.efficientnet_b3(pretrained=True)
# num_ftrs = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(num_ftrs, 50)
# model.to(device)

# #unfreese the classifier head and the last block
# for name, child in model.named_children():
#     if name == 'features':  # "features" contains the main layers
#         for layer_name, layer in list(child.named_children())[-2:]:  # Unfreeze last two blocks
#             for param in layer.parameters():
#                 param.requires_grad = True

# for param in model.classifier[1].parameters():  # Unfreeze the classifier head
#     param.requires_grad = True

# lr = 0.001
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)

# num_epochs = 10
# BATCH_SIZE = batch_size = 8
# lr = 0.001

# Load pre-trained MobileNetV3-Large model
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

# Modify the classifier for 29 classes
# MobileNetV3 has a different classifier structure compared to EfficientNet
# It consists of a sequence of layers including the final classifier
num_classes = 50
model.classifier = nn.Sequential(
    nn.Linear(in_features=960, out_features=1280),
    nn.Hardswish(),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=1280, out_features=num_classes)
)

# Move model to device
model = model.to(device)

# Freeze all layers except the last convolutional block and classifier
# MobileNetV3 uses a different layer structure
for name, param in model.named_parameters():
    param.requires_grad = False  # First freeze all parameters
    
# Unfreeze the last convolutional block (features.12 for MobileNetV3-Large)
for name, param in model.features[12].named_parameters():
    param.requires_grad = True
    
# Unfreeze the classifier
for name, param in model.classifier.named_parameters():
    param.requires_grad = True

# Initialize optimizer with only trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=0.001)

# Loss function remains the same
criterion = nn.CrossEntropyLoss()

# Training hyperparameters
num_epochs = 10
BATCH_SIZE = 8
lr = 0.001

train_losses = []
valid_losses = []
train_acc = []
valid_acc = []
best_valid_loss = float('inf')
from tqdm import tqdm

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    running_train_loss = 0.0
    running_val_loss = 0.0
    correct_train = 0
    total_train = 0
    correct_val = 0
    total_val = 0

    for images, labels in tqdm(train_loader, desc = "Training", leave = True):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc = "Validating", leave = True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_train_loss = running_train_loss / len(train_loader)
    epoch_val_loss = running_val_loss / len(valid_loader)
    train_losses.append(epoch_train_loss)
    valid_losses.append(epoch_val_loss)

    epoch_train_acc = correct_train / total_train
    epoch_val_acc = correct_val / total_val
    train_acc.append(epoch_train_acc)
    valid_acc.append(epoch_val_acc)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')

    if epoch_val_loss < best_valid_loss:
        best_valid_loss = epoch_val_loss
        torch.save(model.state_dict(), 'best_model_FT_Last2Layer.pth')
        print(f"Model weights saved for epoch {epoch + 1} with validation loss: {best_valid_loss:.4f}")

    # Clear CUDA cache
    torch.cuda.empty_cache()

print(f"Batch size: {BATCH_SIZE}, Learning rate: {lr}, Number of epochs: {num_epochs}")
print(f"Best training accuracy: {max(train_acc):.2f}%")
print(f"Best validation accuracy: {max(valid_acc):.2f}%")
print("Plots for current hyperparameter combination:")
epochs_range = range(1, num_epochs + 1)
plt.figure(figsize=(8, 6))
# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, valid_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc, label='Train Accuracy')
plt.plot(epochs_range, valid_acc, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

