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


label_dict = {
    0: "Northern_goshawk",
    1: "Eurasian_sparrowhawk",
    2: "Marsh_warbler",
    3: "Acrocephalus",
    4: "Common_reed_warbler",
    5: "Common_sandpiper",
    6: "Mandarin_duck",
    7: "Wood_duck",
    8: "Razorbill",
    9: "Common_kingfisher",
    10: "Northern_pintail",
    11: "Northern_shoveler",
    12: "Eurasian_teal",
    13: "Mallard",
    14: "Greylag_goose",
    15: "Common_swift",
    16: "Golden_eagle",
    17: "Short-eared_owl",
    18: "Long-eared_owl",
    19: "Little_owl",
    20: "Common_pochard",
    21: "Bohemian_waxwing",
    22: "Eurasian_bittern",
    23: "Dunlin",
    24: "Red_knot",
    25: "Long-tailed_nightjar",
    26: "European_nightjar",
    27: "Common_linnet",
    28: "European_goldfinch",
    29: "Little_ringed_plover",
    30: "Australian_wood_duck",
    31: "European_greenfinch",
    32: "Western_marsh_harrier",
    33: "Hen_harrier",
    34: "Great_spotted_woodpecker",
    35: "Eurasian_jay",
    36: "Eurasian_oystercatcher",
    37: "Common_gull",
    38: "Black-tailed_gull",
    39: "Lesser_black-backed_gull",
    40: "Red_kite",
    41: "Common_pheasant",
    42: "Greater_flamingo",
    43: "European_green_woodpecker",
    44: "Grey_plover",
    45: "Bornean_peacock-pheasant",
    46: "Tawny_owl",
    47: "Western_capercaillie",
    48: "Common_redshank",
    49: "Barn_owl"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.efficientnet_b3(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 50)

model_weights_path = './image/feather_image/best_model_FT_feather.pth'
model.load_state_dict(torch.load(model_weights_path, map_location=device))

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((240, 40)),
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
