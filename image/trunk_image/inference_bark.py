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

import torchvision.models as models
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
num_classes = 29
model.classifier = nn.Sequential(
    nn.Linear(in_features=960, out_features=1280),
    nn.Hardswish(),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=1280, out_features=num_classes)
)


model_weights_path = './image/trunk_image/best_model_FT_Last2Layer.pth'
model.load_state_dict(torch.load(model_weights_path, map_location=device))

label_dict = {
#     'Acer palmatum'                     'Magnolia liliflora Desr'
# 'Aesculus chinensis'                 Malushalliana
# 'Albizia julibrissin'               'Metasequoia glyptostroboides'
# 'Camptotheca acuminata'             'Michelia chapensis'
# 'Cedrus deodara'                    'Osmanthus fragrans'
# 'Celtis sinensis'                   'Photinia serratifolia'
# 'Cinnamomum camphora (Linn) Presl'   Platanus
# 'Elaeocarpus decipiens'             'Populus L'
# 'Flowering cherry'                  'Prunus cerasifera f. atropurpurea'
# 'Ginkgo biloba'                     'Salix babylonica'
# 'Koelreuteria paniculata'           'Sapindus saponaria'
# 'Lagerstroemia indica'              'Styphnolobium japonicum'
# 'Liquidambar formosana'             'Triadica sebifera'
# 'Liriodendron chinense'             'Zelkova serrata'
# 'Magnolia grandiflora L'


    0: "Acer palmatum",
    1: "Aesculus chinensis",
    2: "Albizia julibrissin",
    3: "Camptotheca acuminata",
    4: "Cedrus deodara",
    5: "Celtis sinensis",
    6: "Cinnamomum camphora (Linn) Presl",
    7: "Elaeocarpus decipiens",
    8: "Flowering cherry",
    9: "Ginkgo biloba",
    10: "Koelreuteria paniculata",
    11: "Lagerstroemia indica",
    12: "Liquidambar formosana",
    13: "Liriodendron chinense",
    14: "Magnolia grandiflora L",
    15: "Magnolia liliiflora Desr",
    16: "Malushalliana",
    17: "Metasequoia glyptostroboides",
    18: "Michelia chapensis",
    19: "Osmanthus fragrans",
    20: "Photinia serratifolia",
    21: "Platanus",
    22: "Populus L",
    23: "Prunus cerasifera f. atropurpurea",
    24: "Salix babylonica",
    25: "Sapindus saponaria",
    26: "Styphnolobium japonicum",
    27: "Triadica sebifera",
    28: "Zelkova serrata"

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



def get_species_from_trunk(image_path):
    image = preprocess_image(image_path)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = label_dict[predicted.item()]
        return label








