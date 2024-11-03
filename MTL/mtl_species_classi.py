import cv2
import numpy as np
import pandas as pd
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

def mtl_species_classi(file_path):
    # Check if GPU is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    species = ['asbfly',
    'ashdro1',
    'ashpri1',
    'ashwoo2',
    'asikoe2',
    'asiope1',
    'aspfly1',
    'aspswi1',
    'barfly1',
    'barswa',
    'bcnher',
    'bkcbul1',
    'bkrfla1',
    'bkskit1',
    'bkwsti',
    'bladro1',
    'blaeag1',
    'blakit1',
    'blhori1',
    'blnmon1',
    'blrwar1',
    'bncwoo3',
    'brakit1',
    'brasta1',
    'brcful1',
    'brfowl1',
    'brnhao1',
    'brnshr',
    'brodro1',
    'brwjac1',
    'brwowl1',
    'btbeat1',
    'bwfshr1',
    'categr',
    'chbeat1',
    'cohcuc1',
    'comfla1',
    'comgre',
    'comior1',
    'comkin1',
    'commoo3',
    'commyn',
    'compea',
    'comros',
    'comsan',
    'comtai1',
    'copbar1',
    'crbsun2',
    'cregos1',
    'crfbar1',
    'crseag1',
    'dafbab1',
    'darter2',
    'eaywag1',
    'emedov2',
    'eucdov',
    'eurbla2',
    'eurcoo',
    'forwag1',
    'gargan',
    'gloibi',
    'goflea1',
    'graher1',
    'grbeat1',
    'grecou1',
    'greegr',
    'grefla1',
    'grehor1',
    'grejun2',
    'grenig1',
    'grewar3',
    'grnsan',
    'grnwar1',
    'grtdro1',
    'gryfra',
    'grynig2',
    'grywag',
    'gybpri1',
    'gyhcaf1',
    'heswoo1',
    'hoopoe',
    'houcro1',
    'houspa',
    'inbrob1',
    'indpit1',
    'indrob1',
    'indrol2',
    'indtit1',
    'ingori1',
    'inpher1',
    'insbab1',
    'insowl1',
    'integr',
    'isbduc1',
    'jerbus2',
    'junbab2',
    'junmyn1',
    'junowl1',
    'kenplo1',
    'kerlau2',
    'labcro1',
    'laudov1',
    'lblwar1',
    'lesyel1',
    'lewduc1',
    'lirplo',
    'litegr',
    'litgre1',
    'litspi1',
    'litswi1',
    'lobsun2',
    'maghor2',
    'malpar1',
    'maltro1',
    'malwoo1',
    'marsan',
    'mawthr1',
    'moipig1',
    'nilfly2',
    'niwpig1',
    'nutman',
    'orihob2',
    'oripip1',
    'pabflo1',
    'paisto1',
    'piebus1',
    'piekin1',
    'placuc3',
    'plaflo1',
    'plapri1',
    'plhpar1',
    'pomgrp2',
    'purher1',
    'pursun3',
    'pursun4',
    'purswa3',
    'putbab1',
    'redspu1',
    'rerswa1',
    'revbul',
    'rewbul',
    'rewlap1',
    'rocpig',
    'rorpar',
    'rossta2',
    'rufbab3',
    'ruftre2',
    'rufwoo2',
    'rutfly6',
    'sbeowl1',
    'scamin3',
    'shikra1',
    'smamin1',
    'sohmyn1',
    'spepic1',
    'spodov',
    'spoowl1',
    'sqtbul1',
    'stbkin1',
    'sttwoo1',
    'thbwar1',
    'tibfly3',
    'tilwar1',
    'vefnut1',
    'vehpar1',
    'wbbfly1',
    'wemhar1',
    'whbbul2',
    'whbsho3',
    'whbtre1',
    'whbwag1',
    'whbwat1',
    'whbwoo2',
    'whcbar1',
    'whiter2',
    'whrmun',
    'whtkin2',
    'woosan',
    'wynlau1',
    'yebbab1',
    'yebbul3',
    'zitcis1'] ; num_species_classes = len(species)

    common_names = {'wbbfly1': 'White-bellied Blue Flycatcher',
    'plapri1': 'Plain Prinia',
    'isbduc1': 'Indian Spot-billed Duck',
    'lblwar1': 'Large-billed Leaf Warbler',
    'junowl1': 'Jungle Owlet',
    'blnmon1': 'Black-naped Monarch',
    'comros': 'Common Rosefinch',
    'compea': 'Indian Peafowl',
    'whbwag1': 'White-browed Wagtail',
    'whbwat1': 'White-breasted Waterhen',
    'grejun2': 'Gray Junglefowl',
    'copbar1': 'Coppersmith Barbet',
    'plaflo1': 'Nilgiri Flowerpecker',
    'sqtbul1': 'Square-tailed Bulbul',
    'woosan': 'Wood Sandpiper',
    'yebbul3': 'Yellow-browed Bulbul',
    'aspfly1': 'Indian Paradise-Flycatcher',
    'mawthr1': 'Malabar Whistling-Thrush',
    'lobsun2': "Loten's Sunbird",
    'lesyel1': 'Lesser Yellownape',
    'eucdov': 'Eurasian Collared-Dove',
    'commyn': 'Common Myna',
    'grynig2': 'Jungle Nightjar',
    'ashwoo2': 'Ashy Woodswallow',
    'hoopoe': 'Eurasian Hoopoe',
    'litswi1': 'Little Swift',
    'cregos1': 'Crested Goshawk',
    'vehpar1': 'Vernal Hanging-Parrot',
    'tibfly3': "Tickell's Blue Flycatcher",
    'bkwsti': 'Black-winged Stilt',
    'litegr': 'Little Egret',
    'crseag1': 'Crested Serpent-Eagle',
    'oripip1': 'Paddyfield Pipit',
    'insowl1': 'Indian Scops-Owl',
    'eaywag1': 'Western Yellow Wagtail',
    'brwowl1': 'Brown Wood-Owl',
    'zitcis1': 'Zitting Cisticola',
    'inbrob1': 'Indian Blue Robin',
    'ashpri1': 'Ashy Prinia',
    'marsan': 'Marsh Sandpiper',
    'houcro1': 'House Crow',
    'houspa': 'House Sparrow',
    'blhori1': 'Black-hooded Oriole',
    'asbfly': 'Asian Brown Flycatcher',
    'piebus1': 'Pied Bushchat',
    'litgre1': 'Little Grebe',
    'bkcbul1': 'Flame-throated Bulbul',
    'rorpar': 'Rose-ringed Parakeet',
    'piekin1': 'Pied Kingfisher',
    'spodov': 'Spotted Dove',
    'indrob1': 'Indian Robin',
    'aspswi1': 'Asian Palm-Swift',
    'blaeag1': 'Black Eagle',
    'eurcoo': 'Eurasian Coot',
    'malwoo1': 'Malabar Woodshrike',
    'cohcuc1': 'Common Hawk-Cuckoo',
    'ashdro1': 'Ashy Drongo',
    'comtai1': 'Common Tailorbird',
    'categr': 'Cattle Egret',
    'whbbul2': 'White-browed Bulbul',
    'moipig1': 'Mountain Imperial-Pigeon',
    'commoo3': 'Eurasian Moorhen',
    'purswa3': 'Gray-headed Swamphen',
    'insbab1': 'Indian Scimitar-Babbler',
    'smamin1': 'Small Minivet',
    'whcbar1': 'White-cheeked Barbet',
    'grefla1': 'Greater Flameback',
    'eurbla2': 'Indian Blackbird',
    'plhpar1': 'Plum-headed Parakeet',
    'btbeat1': 'Blue-tailed Bee-eater',
    'paisto1': 'Painted Stork',
    'thbwar1': 'Thick-billed Warbler',
    'goflea1': 'Golden-fronted Leafbird',
    'lewduc1': 'Lesser Whistling-Duck',
    'indrol2': 'Indian Roller',
    'grywag': 'Gray Wagtail',
    'wemhar1': 'Eurasian Marsh-Harrier',
    'rossta2': 'Rosy Starling',
    'dafbab1': 'Dark-fronted Babbler',
    'ruftre2': 'Rufous Treepie',
    'brasta1': 'Brahminy Starling',
    'rufbab3': 'Rufous Babbler',
    'comior1': 'Common Iora',
    'labcro1': 'Large-billed Crow',
    'emedov2': 'Asian Emerald Dove',
    'brnhao1': 'Brown Boobook',
    'blrwar1': "Blyth's Reed Warbler",
    'pursun4': 'Purple Sunbird',
    'crbsun2': 'Crimson-backed Sunbird',
    'gryfra': 'Gray Francolin',
    'kerlau2': 'Palani Laughingthrush',
    'scamin3': 'Orange Minivet',
    'comfla1': 'Common Flameback',
    'graher1': 'Gray Heron',
    'spepic1': 'Speckled Piculet',
    'kenplo1': 'Kentish Plover',
    'lirplo': 'Little Ringed Plover',
    'grbeat1': 'Green Bee-eater',
    'bkskit1': 'Black-winged Kite',
    'junbab2': 'Jungle Babbler',
    'gargan': 'Garganey',
    'purher1': 'Purple Heron',
    'grnsan': 'Green Sandpiper',
    'junmyn1': 'Jungle Myna',
    'bkrfla1': 'Black-rumped Flameback',
    'laudov1': 'Laughing Dove',
    'litspi1': 'Little Spiderhunter',
    'rewlap1': 'Red-wattled Lapwing',
    'rufwoo2': 'Rufous Woodpecker',
    'rutfly6': 'Rusty-tailed Flycatcher',
    'asiope1': 'Asian Openbill',
    'tilwar1': "Tickell's Leaf Warbler",
    'placuc3': 'Gray-bellied Cuckoo',
    'barswa': 'Barn Swallow',
    'comsan': 'Common Sandpiper',
    'sohmyn1': 'Southern Hill Myna',
    'whbsho3': 'White-bellied Sholakili',
    'shikra1': 'Shikra',
    'heswoo1': 'Heart-spotted Woodpecker',
    'brfowl1': 'Brown Fish-Owl',
    'rerswa1': 'Red-rumped Swallow',
    'yebbab1': 'Yellow-billed Babbler',
    'indtit1': 'Indian Yellow Tit',
    'rocpig': 'Rock Pigeon',
    'bcnher': 'Black-crowned Night-Heron',
    'revbul': 'Red-vented Bulbul',
    'maghor2': 'Malabar Gray Hornbill',
    'comkin1': 'Common Kingfisher',
    'inpher1': 'Indian Pond-Heron',
    'malpar1': 'Malabar Parakeet',
    'forwag1': 'Forest Wagtail',
    'whbtre1': 'White-bellied Treepie',
    'barfly1': 'Black-and-orange Flycatcher',
    'maltro1': 'Malabar Trogon',
    'grenig1': 'Great Eared-Nightjar',
    'indpit1': 'Indian Pitta',
    'bladro1': 'Black Drongo',
    'niwpig1': 'Nilgiri Wood-Pigeon',
    'vefnut1': 'Velvet-fronted Nuthatch',
    'whtkin2': 'White-throated Kingfisher',
    'nutman': 'Scaly-breasted Munia',
    'crfbar1': 'Malabar Barbet',
    'grtdro1': 'Greater Racket-tailed Drongo',
    'comgre': 'Common Greenshank',
    'bwfshr1': 'Bar-winged Flycatcher-shrike',
    'whbwoo2': 'White-bellied Woodpecker',
    'pabflo1': 'Pale-billed Flowerpecker',
    'stbkin1': 'Stork-billed Kingfisher',
    'grnwar1': 'Green Warbler',
    'sttwoo1': 'Streak-throated Woodpecker',
    'jerbus2': "Jerdon's Bushlark",
    'nilfly2': 'Nilgiri Flycatcher',
    'grecou1': 'Greater Coucal',
    'darter2': 'Oriental Darter',
    'rewbul': 'Red-whiskered Bulbul',
    'gloibi': 'Glossy Ibis',
    'spoowl1': 'Spotted Owlet',
    'bncwoo3': 'Brown-capped Pygmy Woodpecker',
    'orihob2': 'Oriental Honey-buzzard',
    'gyhcaf1': 'Gray-headed Canary-Flycatcher',
    'sbeowl1': 'Spot-bellied Eagle-Owl',
    'putbab1': 'Puff-throated Babbler',
    'wynlau1': 'Wayanad Laughingthrush',
    'greegr': 'Great Egret',
    'pursun3': 'Purple-rumped Sunbird',
    'pomgrp2': 'Gray-fronted Green-Pigeon',
    'whiter2': 'Whiskered Tern',
    'brakit1': 'Brahminy Kite',
    'grewar3': 'Greenish Warbler',
    'grehor1': 'Great Hornbill',
    'gybpri1': 'Gray-breasted Prinia',
    'redspu1': 'Red Spurfowl',
    'whrmun': 'White-rumped Munia',
    'blakit1': 'Black Kite',
    'brnshr': 'Brown Shrike',
    'chbeat1': 'Chestnut-headed Bee-eater',
    'integr': 'Intermediate Egret',
    'brwjac1': 'Bronze-winged Jacana',
    'asikoe2': 'Asian Koel',
    'brcful1': 'Brown-cheeked Fulvetta',
    'ingori1': 'Indian Golden Oriole',
    'brodro1': 'Bronzed Drongo'}

    orders = ['Charadriiformes',
    'Gruiformes',
    'Accipitriformes',
    'Podicipediformes',
    'Trogoniformes',
    'Bucerotiformes',
    'Caprimulgiformes',
    'Coraciiformes',
    'Strigiformes',
    'Anseriformes',
    'Galliformes',
    'Passeriformes',
    'Pelecaniformes',
    'Piciformes',
    'Columbiformes',
    'Psittaciformes',
    'Ciconiiformes',
    'Suliformes',
    'Cuculiformes'] ; num_order_classes = len(orders)

    family = ['Scolopacidae (Sandpipers and Allies)',
    'Passeridae (Old World Sparrows)',
    'Paridae (Tits, Chickadees, and Titmice)',
    'Laniidae (Shrikes)',
    'Stenostiridae (Fairy Flycatchers)',
    'Pycnonotidae (Bulbuls)',
    'Sturnidae (Starlings)',
    'Phasianidae (Pheasants, Grouse, and Allies)',
    'Meropidae (Bee-eaters)',
    'Estrildidae (Waxbills and Allies)',
    'Cisticolidae (Cisticolas and Allies)',
    'Dicruridae (Drongos)',
    'Recurvirostridae (Stilts and Avocets)',
    'Vangidae (Vangas, Helmetshrikes, and Allies)',
    'Pittidae (Pittas)',
    'Cuculidae (Cuckoos)',
    'Columbidae (Pigeons and Doves)',
    'Motacillidae (Wagtails and Pipits)',
    'Coraciidae (Rollers)',
    'Anhingidae (Anhingas)',
    'Bucerotidae (Hornbills)',
    'Megalaimidae (Asian Barbets)',
    'Alaudidae (Larks)',
    'Alcedinidae (Kingfishers)',
    'Podicipedidae (Grebes)',
    'Rallidae (Rails, Gallinules, and Coots)',
    'Corvidae (Crows, Jays, and Magpies)',
    'Sittidae (Nuthatches)',
    'Oriolidae (Old World Orioles)',
    'Ardeidae (Herons, Egrets, and Bitterns)',
    'Turdidae (Thrushes and Allies)',
    'Leiothrichidae (Laughingthrushes and Allies)',
    'Picidae (Woodpeckers)',
    'Threskiornithidae (Ibises and Spoonbills)',
    'Upupidae (Hoopoes)',
    'Dicaeidae (Flowerpeckers)',
    'Strigidae (Owls)',
    'Aegithinidae (Ioras)',
    'Apodidae (Swifts)',
    'Chloropseidae (Leafbirds)',
    'Caprimulgidae (Nightjars and Allies)',
    'Laridae (Gulls, Terns, and Skimmers)',
    'Nectariniidae (Sunbirds and Spiderhunters)',
    'Phylloscopidae (Leaf Warblers)',
    'Pellorneidae (Ground Babblers and Allies)',
    'Campephagidae (Cuckooshrikes)',
    'Fringillidae (Finches, Euphonias, and Allies)',
    'Timaliidae (Tree-Babblers, Scimitar-Babblers, and Allies)',
    'Charadriidae (Plovers and Lapwings)',
    'Psittaculidae (Old World Parrots)',
    'Accipitridae (Hawks, Eagles, and Kites)',
    'Artamidae (Woodswallows, Bellmagpies, and Allies)',
    'Monarchidae (Monarch Flycatchers)',
    'Jacanidae (Jacanas)',
    'Acrocephalidae (Reed Warblers and Allies)',
    'Anatidae (Ducks, Geese, and Waterfowl)',
    'Trogonidae (Trogons)',
    'Muscicapidae (Old World Flycatchers)',
    'Ciconiidae (Storks)',
    'Hirundinidae (Swallows)'] ; num_family_classes = len(family)


    class CFG:    
        # Input image size and batch size
        img_size = [224, 224]
        
        # Audio duration, sample rate, and length
        duration = 5 # second
        sample_rate = 32000
        audio_len = duration*sample_rate
        
        # Short-Time Fourier Transform(STFT) parameters
        nfft = 2028       #The number of points in the FFT. A higher nfft value can provide better frequency resolution but will require more computational power and time.
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
            self.species_fc = nn.Linear(1280, 512)  # Task-specific FC layer for species
            self.species_class = nn.Linear(512, num_species_classes)  # Classification layer for species

            # Order task layers
            self.order_fc = nn.Linear(1280, 512)  # Task-specific FC layer for order
            self.order_class = nn.Linear(512, num_order_classes)  # Classification layer for order

            # Family task layers
            self.family_fc = nn.Linear(1280, 512)  # Task-specific FC layer for family
            self.family_class = nn.Linear(512, num_family_classes)  # Classification layer for family

            # Activation and dropout
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            features = self.backbone(x)  # Output shape: (batch_size, 1280)
            pooled_features = features.view(features.size(0), -1)  # Flatten: (batch_size, 1280)

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
    state_dict = torch.load("MTL/best_species_model.weights.pth", map_location=device)
    model_infer.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model_infer.eval()

    def build_decoder_inference(dim=5*32000):
        def get_audio(filepath):
            audio, sr = librosa.load(filepath, sr=CFG.sample_rate)  # Load audio file with fixed sampling rate (CFG.sample_rate)
            if len(audio.shape) > 1:  # stereo -> mono
                audio = librosa.to_mono(audio)
            return torch.tensor(audio)
        
        def create_frames(audio, duration=5, sr=32000):
            frame_size = int(duration*sr)
            # Pad the end of the audio tensor so it's divisible by frame_size
            pad_length = frame_size - (audio.size(0) % frame_size)
            if pad_length < frame_size:  # Only pad if there's a remainder
                audio = F.pad(audio, (0, pad_length))

            # Reshape audio to create frames
            frames = audio.view(-1, frame_size)  # shape: [num_frames, frame_size]
        
            return frames

        def apply_preproc(spec):
            # Standardize
            mean = np.mean(spec)
            std = np.std(spec)
            spec = (spec - mean) / std if std != 0 else spec - mean

            # Normalize using Min-Max
            min_val = np.min(spec)
            max_val = np.max(spec)
            spec = (spec - min_val) / (max_val - min_val) if max_val != min_val else spec - min_val
            return spec

        def decode(path):
            # Load audio file
            audio = get_audio(path)
            # Crop or pad audio to keep a fixed length
            audio_frames = create_frames(audio)
            spectrograms = []
            for frame in audio_frames:
                spec = librosa.feature.melspectrogram(y=frame.numpy(), sr=CFG.sample_rate, n_fft=CFG.nfft, hop_length=CFG.hop_length, n_mels=CFG.img_size[0])
                spec = librosa.power_to_db(spec, ref=np.max)
                spec = apply_preproc(spec)
                spec = np.tile(spec[..., None], [1, 1, 3])
                # Resize the spectrogram to the desired shape
                spec = cv2.resize(spec, (CFG.img_size[1], CFG.img_size[0]))
                spec = np.reshape(spec, [*CFG.img_size, 3])
                spectrograms.append(spec)
            return torch.tensor(spectrograms)

        return decode

    decode_fn = build_decoder_inference()
    audio_data = decode_fn(file_path)
    audio_data = audio_data.permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        species_preds, _, _ = model_infer(audio_data)  # Only take the species output
        species_probs = F.softmax(species_preds, dim=1)  # Apply softmax to get probabilities
            
        # Move the predictions back to CPU and convert to NumPy array
        frame_preds = species_probs.cpu().numpy()

    sp = [species[np.argmax(frame)] for frame in frame_preds]
    from collections import Counter
    species_count = Counter(sp)
    final_pred = species_count.most_common(1)[0][0]
    common_name = common_names[final_pred]
    return common_name

print(mtl_species_classi("MTL/1003342351.ogg"))
