import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
import torch.optim as optim
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchaudio
from torchaudio.transforms import Resample
import torchaudio.transforms as T
import warnings
warnings.filterwarnings("ignore")
from audio.species.data import species, num_species_classes, common_names, orders, num_order_classes, family, num_family_classes
from PIL import Image
import io

def mtl_species_classi(file_path):
    # Check if GPU is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class CFG:    
        # Input image size and batch size
        img_size = [224, 224]
        
        # Audio duration, sample rate, and length
        duration = 5 # second
        sample_rate = 32000
        audio_len = duration * sample_rate
        
        # Short-Time Fourier Transform(STFT) parameters
        nfft = 2028       # The number of points in the FFT
        window = 2048
        hop_length = audio_len // (img_size[1] - 1)
        fmin = 20
        fmax = 16000
        num_classes = num_species_classes

    class MultiTaskModel(nn.Module):
        def __init__(self, num_species_classes, num_order_classes, num_family_classes):
            super(MultiTaskModel, self).__init__()
            
            # Pretrained EfficientNet backbone
            self.backbone = models.efficientnet_v2_s(pretrained=True)
            self.backbone.classifier = nn.Identity()  # Remove the final classification layer
            
            # Species task layers
            self.species_fc = nn.Linear(1280, 512)
            self.species_class = nn.Linear(512, num_species_classes)

            # Order task layers
            self.order_fc = nn.Linear(1280, 512)
            self.order_class = nn.Linear(512, num_order_classes)

            # Family task layers
            self.family_fc = nn.Linear(1280, 512)
            self.family_class = nn.Linear(512, num_family_classes)

            # Activation and dropout
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            features = self.backbone(x)
            pooled_features = features.view(features.size(0), -1)

            # Species branch
            species_output = self.relu(self.species_fc(pooled_features))
            species_output = self.dropout(species_output)
            species_output = self.species_class(species_output)

            # Order branch
            order_output = self.relu(self.order_fc(pooled_features))
            order_output = self.dropout(order_output)
            order_output = self.order_class(order_output)

            # Family branch
            family_output = self.relu(self.family_fc(pooled_features))
            family_output = self.dropout(family_output)
            family_output = self.family_class(family_output)

            return species_output, order_output, family_output

    model_infer = MultiTaskModel(num_species_classes, num_order_classes, num_family_classes).to(device)

    # Load the state dictionary from the file
    state_dict = torch.load("./audio/species/best_species_model.weights_new.pth", map_location=device)
    model_infer.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model_infer.eval()

    def build_decoder_inference(dim=5*32000):
        def get_audio(filepath):
            audio, sr = librosa.load(filepath, sr=CFG.sample_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            return torch.tensor(audio)
        
        def create_frames(audio, duration=5, sr=32000):
            frame_size = int(duration * sr)
            pad_length = frame_size - (audio.size(0) % frame_size)
            if pad_length < frame_size:
                audio = F.pad(audio, (0, pad_length))
            frames = audio.view(-1, frame_size)
            return frames

        def apply_preproc(spec):
            mean = np.mean(spec)
            std = np.std(spec)
            spec = (spec - mean) / std if std != 0 else spec - mean

            min_val = np.min(spec)
            max_val = np.max(spec)
            spec = (spec - min_val) / (max_val - min_val) if max_val != min_val else spec - min_val
            return spec

        def decode(path):
            audio = get_audio(path)
            audio_frames = create_frames(audio)
            spectrograms = []
            for frame in audio_frames:
                spec = librosa.feature.melspectrogram(y=frame.numpy(), sr=CFG.sample_rate, n_fft=CFG.nfft, hop_length=CFG.hop_length, n_mels=CFG.img_size[0])
                spec = librosa.power_to_db(spec, ref=np.max)
                spec = apply_preproc(spec)
                spec = np.tile(spec[..., None], [1, 1, 3])
                spec_image = Image.fromarray((spec * 255).astype(np.uint8)).resize(CFG.img_size, Image.LANCZOS)
                spec_image = np.array(spec_image) / 255.0
                spectrograms.append(spec_image)
            return torch.tensor(spectrograms)

        return decode
    
    def plot_mel_spec(audio_data):
        # Convert the first channel to 2D if it's 3D
        if audio_data[0].shape[0] == 3:
            data_to_plot = audio_data[0][0]
        else:
            data_to_plot = audio_data[0]

        # Plot the mel spectrogram
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(data_to_plot, cmap='coolwarm')
        ax.axis('off')
        plt.show()
        return fig

    decode_fn = build_decoder_inference()
    audio_data = decode_fn(file_path)
    audio_data = audio_data.permute(0, 3, 1, 2).to(device)
    plot = plot_mel_spec(audio_data)

    with torch.no_grad():
        species_preds, _, _ = model_infer(audio_data)
        species_probs = F.softmax(species_preds, dim=1)
            
        frame_preds = species_probs.cpu().numpy()

    sp = [species[np.argmax(frame)] for frame in frame_preds]
    from collections import Counter
    species_count = Counter(sp)
    final_pred = species_count.most_common(1)[0][0]
    common_name = common_names[final_pred]
    return common_name, plot
# def mtl_species_classi(file_path):
#     # Check if GPU is available and set the device accordingly
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     class CFG:    
#         # Input image size and batch size
#         img_size = [224, 224]
        
#         # Audio duration, sample rate, and length
#         duration = 5 # second
#         sample_rate = 32000
#         audio_len = duration*sample_rate
        
#         # Short-Time Fourier Transform(STFT) parameters
#         nfft = 2028       #The number of points in the FFT. A higher nfft value can provide better frequency resolution but will require more computational power and time.
#         window = 2048
#         hop_length = audio_len // (img_size[1] - 1)
#         fmin = 20
#         fmax = 16000
#         num_classes = num_species_classes

#     class MultiTaskModel(nn.Module):
#         def __init__(self, num_species_classes, num_order_classes, num_family_classes):
#             super(MultiTaskModel, self).__init__()
            
#             # Pretrained EfficientNet backbone
#             self.backbone = models.efficientnet_v2_s(pretrained=True)
#             self.backbone.classifier = nn.Identity()  # Remove the final classification layer
            
#             # Species task layers
#             self.species_fc = nn.Linear(1280, 512)  # Task-specific FC layer for species
#             self.species_class = nn.Linear(512, num_species_classes)  # Classification layer for species

#             # Order task layers
#             self.order_fc = nn.Linear(1280, 512)  # Task-specific FC layer for order
#             self.order_class = nn.Linear(512, num_order_classes)  # Classification layer for order

#             # Family task layers
#             self.family_fc = nn.Linear(1280, 512)  # Task-specific FC layer for family
#             self.family_class = nn.Linear(512, num_family_classes)  # Classification layer for family

#             # Activation and dropout
#             self.relu = nn.ReLU()
#             self.dropout = nn.Dropout(0.3)

#         def forward(self, x):
#             features = self.backbone(x)  # Output shape: (batch_size, 1280)
#             pooled_features = features.view(features.size(0), -1)  # Flatten: (batch_size, 1280)

#             # Species branch
#             species_output = self.relu(self.species_fc(pooled_features))
#             species_output = self.dropout(species_output)
#             species_output = self.species_class(species_output)

#             # Order branch
#             order_output = self.relu(self.order_fc(pooled_features))
#             order_output = self.dropout(order_output)
#             order_output = self.order_class(order_output)

#             # Family branch
#             family_output = self.relu(self.family_fc(pooled_features))
#             family_output = self.dropout(family_output)
#             family_output = self.family_class(family_output)

#             return species_output, order_output, family_output

#     model_infer = MultiTaskModel(num_species_classes, num_order_classes, num_family_classes).to(device)

#     # Load the state dictionary from the file
#     state_dict = torch.load("./audio/species/best_species_model.weights_new.pth", map_location=device)
#     model_infer.load_state_dict(state_dict)

#     # Set the model to evaluation mode
#     model_infer.eval()

#     def build_decoder_inference(dim=5*32000):
#         def get_audio(filepath):
#             audio, sr = librosa.load(filepath, sr=CFG.sample_rate)  # Load audio file with fixed sampling rate (CFG.sample_rate)
#             if len(audio.shape) > 1:  # stereo -> mono
#                 audio = librosa.to_mono(audio)
#             return torch.tensor(audio)
        
#         def create_frames(audio, duration=5, sr=32000):
#             frame_size = int(duration*sr)
#             # Pad the end of the audio tensor so it's divisible by frame_size
#             pad_length = frame_size - (audio.size(0) % frame_size)
#             if pad_length < frame_size:  # Only pad if there's a remainder
#                 audio = F.pad(audio, (0, pad_length))

#             # Reshape audio to create frames
#             frames = audio.view(-1, frame_size)  # shape: [num_frames, frame_size]
        
#             return frames

#         def apply_preproc(spec):
#             # Standardize
#             mean = np.mean(spec)
#             std = np.std(spec)
#             spec = (spec - mean) / std if std != 0 else spec - mean

#             # Normalize using Min-Max
#             min_val = np.min(spec)
#             max_val = np.max(spec)
#             spec = (spec - min_val) / (max_val - min_val) if max_val != min_val else spec - min_val
#             return spec

#         def decode(path):
#             # Load audio file
#             audio = get_audio(path)
#             # Crop or pad audio to keep a fixed length
#             audio_frames = create_frames(audio)
#             spectrograms = []
#             for frame in audio_frames:
#                 spec = librosa.feature.melspectrogram(y=frame.numpy(), sr=CFG.sample_rate, n_fft=CFG.nfft, hop_length=CFG.hop_length, n_mels=CFG.img_size[0])
#                 spec = librosa.power_to_db(spec, ref=np.max)
#                 spec = apply_preproc(spec)
#                 spec = np.tile(spec[..., None], [1, 1, 3])
#                 # Resize the spectrogram to the desired shape
#                 spec = cv2.resize(spec, (CFG.img_size[1], CFG.img_size[0]))
#                 spec = np.reshape(spec, [*CFG.img_size, 3])
#                 spectrograms.append(spec)
#             return torch.tensor(spectrograms)

#         return decode
    
#     def plot_mel_spec(audio_data):
#         # Convert the first channel to 2D if it's 3D
#         if audio_data[0].shape[0] == 3:
#             data_to_plot = audio_data[0][0]  # Take the first channel
#         else:
#             data_to_plot = audio_data[0]

#         # Plot the mel spectrogram
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.imshow(data_to_plot, cmap='coolwarm')
#         ax.axis('off')
#         plt.show()
#         return fig

#     decode_fn = build_decoder_inference()
#     audio_data = decode_fn(file_path)
#     audio_data = audio_data.permute(0, 3, 1, 2).to(device)
#     plot = plot_mel_spec(audio_data)

#     with torch.no_grad():
#         species_preds, _, _ = model_infer(audio_data)  # Only take the species output
#         species_probs = F.softmax(species_preds, dim=1)  # Apply softmax to get probabilities
            
#         # Move the predictions back to CPU and convert to NumPy array
#         frame_preds = species_probs.cpu().numpy()

#     sp = [species[np.argmax(frame)] for frame in frame_preds]
#     from collections import Counter
#     species_count = Counter(sp)
#     final_pred = species_count.most_common(1)[0][0]
#     common_name = common_names[final_pred]
#     return common_name, plot
