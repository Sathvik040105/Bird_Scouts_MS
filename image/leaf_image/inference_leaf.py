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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import os
from PIL import Image

import torchvision.models as models
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
num_classes = 50
model.classifier = nn.Sequential(
    nn.Linear(in_features=960, out_features=1280),
    nn.Hardswish(),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=1280, out_features=num_classes)
)


model_weights_path = './image/leaf_image/best_model_FT_Last2Layer.pth'
model.load_state_dict(torch.load(model_weights_path, map_location=device))

label_dict = {    
    0: 'Acer palmatum',
    1: 'Aesculus chinensis',
    2: 'Albizia julibrissin',
    3: 'Aucuba japonica var. variegata',
    4: 'Buxus sinica var. parvifolia',
    5: 'Camptotheca acuminata',
    6: 'Cedrus deodara',
    7: 'Celtis sinensis',
    8: 'Cinnamomum camphora (Linn) Presl',
    9: 'Elaeocarpus decipiens',
    10: 'Euonymus japonicus',
    11: 'Euonymus japonicus Aureo_marginatus',
    12: 'Flowering cherry',
    13: 'Ginkgo biloba',
    7: 'Celtis sinensis',
    8: 'Cinnamomum camphora (Linn) Presl',
    9: 'Elaeocarpus decipiens',
    14: 'Juniperus chinensis Kaizuca',
    15: 'Koelreuteria paniculata',
    16: 'Lagerstroemia indica',
    17: 'Ligustrum lucidum',
    18: 'Liquidambar formosana',
    19: 'Liriodendron chinense',
    20: 'Llex cornuta',
    21: 'Loropetalum chinense var. rubrum',
    22: 'Magnolia grandiflora L',
    23: 'Magnolia liliflora Desr',
    24: 'Malushalliana',
    25: 'Metasequoia glyptostroboides',
    26: 'Michelia chapensis',
    27: 'Michelia figo (Lour.) Spreng',
    28: 'Nandina domestica',
    29: 'Nerium oleander L',
    30: 'Osmanthus fragrans',
    31: 'Photinia serratifolia',
    32: 'Pinus massoniana Lamb',
    33: 'Pinus parviflora',
    34: 'Pittosporum tobira',
    35: 'Platanus',
    36: 'Platycladus orientalis Beverlevensis',
    37: 'Podocarpus macrophyllus',
    38: 'Populus L',
    39: 'Prunus cerasifera f. atropurpurea',
    40: 'Prunus persica',
    41: 'Salix babylonica',
    42: 'Sapindus saponaria',
    43: 'Styphnolobium japonicum',
    44: 'Triadica sebifera',
    45: 'Zelkova serrata',
    46: 'Magnolia grandiflora L',
    47: 'Magnolia liliflora Desr',
    48: 'Metasequoia glyptostroboides',
    49: 'Michelia chapensis'


}

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = image_path.convert('RGB')
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







