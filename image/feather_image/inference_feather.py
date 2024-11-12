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
    0: "accipiter_gentilis",
    1: "accipiter_nisus",
    2: "acrocephalus_palustris",
    3: "acrocephalus_schoenobaenus",
    4: "acrocephalus_scirpaceus",
    5: "actitis_hypoleucos",
    6: "aix_galericulata",
    7: "aix_sponsa",
    8: "alca_torda",
    9: "alcedo_atthis",
    10: "anas_acuta",
    11: "anas_clypeata",
    12: "anas_crecca",
    13: "anas_platyrhynchos",
    14: "anser_anser",
    15: "apus_apus",
    16: "aquila_chrysaetos",
    17: "asio_flammeus",
    18: "asio_otus",
    19: "athene_noctua",
    20: "aythya_ferina",
    21: "bombycilla_garrulus",
    22: "botaurus_stellaris",
    23: "calidris_alpina",
    24: "calidris_canutus",
    25: "caprimulgus_climacurus",
    26: "caprimulgus_europaeus",
    27: "carduelis_cannabina",
    28: "carduelis_carduelis",
    29: "charadrius_dubius",
    30: "chenonetta_jubata",
    31: "chloris_chloris",
    32: "circus_aeruginosus",
    33: "circus_cyaneus",
    34: "dendrocopos_major",
    35: "garrulus_glandarius",
    36: "haematopus_ostralegus",
    37: "larus_canus",
    38: "larus_crassirostris",
    39: "larus_fuscus",
    40: "milvus_milvus",
    41: "phasianus_colchicus",
    42: "phoenicopterus_roseus",
    43: "picus_viridis",
    44: "pluvialis_squatarola",
    45: "polyplectron_schleiermacheri",
    46: "strix_aluco",
    47: "tetro_urogallus",
    48: "tringa_totanus",
    49: "tyto_alba"
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