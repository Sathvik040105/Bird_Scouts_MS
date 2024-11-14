# Aditya Manjunatha

import torch
import torch.nn as nn
import torch.nn.functional as F#All activation functions are present here
import torch.optim as optim # Optimizer is stored here
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import os
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.efficientnet_b3(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 25)

model_weights_path = './image/bird_image/best_model_FullTrainScratch.pth'
model.load_state_dict(torch.load(model_weights_path, map_location=device))

label_dict = {
    0: "Asian-Green-Bee-Eater",
    1: "Brown-Headed-Barbet",
    2: "Cattle-Egret",
    3: "Common-Kingfisher",
    4: "Common-Myna",
    5: "Common-Rosefinch",
    6: "Common-Tailorbird",
    7: "Coppersmith-Barbet",
    8: "Forest-Wagtail",
    9: "Gray-Wagtail",
    10: "Hoopoe",
    11: "House-Crow",
    12: "Indian-Grey-Hornbill",
    13: "Indian-Peacock",
    14: "Indian-Pitta",
    15: "Indian-Roller",
    16: "Jungle-Babbler",
    17: "Northern-Lapwing",
    18: "Red-Wattled-Lapwing",
    19: "Ruddy-Shelduck",
    20: "Rufous-Treepie",
    21: "Sarus-Crane",
    22: "White-Breasted-Kingfisher",
    23: "White-Breasted-Waterhen",
    24: "White-Wagtail"
}

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = image.convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0) 
    return image



def predict_image_class(image_path):
    image = preprocess_image(image_path)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = label_dict[predicted.item()]
        return label
