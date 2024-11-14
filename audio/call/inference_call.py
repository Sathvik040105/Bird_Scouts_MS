import torch
import torch.nn as nn
import torchaudio
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights
import torchvision.models
import streamlit as st
from io import BytesIO

mel_spec_params = {
    "sample_rate": 32000,
    "n_mels": 128,
    "f_min": 20,
    "f_max": 16000,
    "n_fft": 2048,
    "hop_length": 512,
    "normalized": True,
    "center" : True,
    "pad_mode" : "constant",
    "norm" : "slaney",
    "onesided" : True,
    "mel_scale" : "slaney"
}

label_mapping = {
    0 : 'Call',
    1 : 'Song',
    2 : 'Dawn song',
    3 : 'Non-vocal song',
    4 : 'Duet',
    5 : 'Flight song',
    6 : 'Flight call'
}

top_db = 80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model class
class BirdCallClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BirdCallClassifier, self).__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.efficientnet = torchvision.models.efficientnet_b0(weights=weights)
        self.efficientnet.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        x = x.to(device)  # Ensure input is on GPU
        x = self.efficientnet(x)
        return x

# Load the saved model for inference
def load_model(model_path, num_classes):
    model = BirdCallClassifier(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Function to normalize mel spectrogram
def normalize_melspec(X, eps=1e-6):
    X = X.to(device)  # Move tensor to GPU
    mean = X.mean((1, 2), keepdim=True)
    std = X.std((1, 2), keepdim=True)
    Xstd = (X - mean) / (std + eps)

    norm_min, norm_max = (
        Xstd.min(-1)[0].min(-1)[0],
        Xstd.max(-1)[0].max(-1)[0],
    )
    fix_ind = (norm_max - norm_min) > eps * torch.ones_like((norm_max - norm_min)).to(device)
    V = torch.zeros_like(Xstd).to(device)
    if fix_ind.sum():
        V_fix = Xstd[fix_ind]
        norm_max_fix = norm_max[fix_ind, None, None]
        norm_min_fix = norm_min[fix_ind, None, None]
        V_fix = torch.max(
            torch.min(V_fix, norm_max_fix),
            norm_min_fix,
        )
        V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
        V[fix_ind] = V_fix
    return V

# Function to read and resample wav file
def read_wav(path):
    wav, org_sr = torchaudio.load(BytesIO(path.getvalue()), normalize=True)
    wav = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=mel_spec_params["sample_rate"])
    wav = wav.to(device)  # Move tensor to GPU
    return wav

# Preprocess audio file for inference
def preprocess_audio(audio_path):
    waveform = read_wav(audio_path).to(device)
    db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db).to(device)
    mel_transform = torchaudio.transforms.MelSpectrogram(**mel_spec_params).to(device)
    mel_spectrogram = normalize_melspec(db_transform(mel_transform(waveform))).to(device)
    mel_spectrogram = mel_spectrogram * 255
    
    # Ensure the tensor has 3 channels
    if mel_spectrogram.size(0) == 1:
        mel_spectrogram = mel_spectrogram.repeat(3, 1, 1)
    elif mel_spectrogram.size(0) == 2:
        mel_spectrogram = torch.cat([mel_spectrogram, mel_spectrogram[0:1, :, :]], dim=0)
    elif mel_spectrogram.size(0) != 3:
        raise RuntimeError(f"Unexpected tensor size: {mel_spectrogram.size()}")

    mel_spectrogram = mel_spectrogram.permute(1, 2, 0).cpu().numpy()  # Move back to CPU for numpy
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    log_mel_spectrogram = transform(mel_spectrogram)
    
    log_mel_spectrogram = log_mel_spectrogram.unsqueeze(0)  # Add batch dimension
    
    return log_mel_spectrogram

@st.cache_resource
def get_model():
    model = load_model('./audio/call/bird_call_classifier.pth', 7)
    return model    

# Perform inference on a single audio file
def predict_audio_class(audio_path):
    model = get_model()
    audio_tensor = preprocess_audio(audio_path)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(audio_tensor.to(device))
        _, predicted = torch.max(output, 1)
        
        label = label_mapping[predicted.item()]
        return label
