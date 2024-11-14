import torch
from torchvision.models import detection
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np
from PIL import ImageFont
import torchvision.models as models
import torch
import torch.nn as nn

#Change device to CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"

#Object detection model
def load_model():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Loaded a pretrained FasterCNN model for accurate object detection
    model = detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)#pretrained on COCO
    model.to(device)  # Move model to GPU if available
    model.eval()
    return model, device

#Annotates the image
def annotate_image(image, boxes, scores, labels, coco_classes, confidence_threshold=0.7):
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Use default font, no explicit font size control
    font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        if scores[i] > confidence_threshold and labels[i] == 16:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)
            class_name = coco_classes[labels[i]]
            draw.text((x1, y1 - 20), f"{class_name}: {scores[i]:.2f}", fill=(255, 0, 0), font=font)
    
    return annotated_image

#Function that calls the object detection and annotation
def detect_and_annotate_single_image(image_path, confidence_threshold=0.7):
    """
    Detect objects in an image, annotate the original image with bounding boxes, and return the annotated image
    
    Args:
        image_path (str or Path): Path to the input image
        confidence_threshold (float): Minimum confidence score for detections
        
    Returns:
        PIL.Image: Annotated image
    """
    # Load and preprocess image
    image = image_path
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(image)
    
    # Load model (if not already loaded)
    model, device = load_model()
    img_tensor = img_tensor.to(device)  # Move input to GPU if available
    
    # Get predictions
    with torch.no_grad():
        predictions = model([img_tensor])
    
    # Get the prediction for the image
    pred = predictions[0]
    
    # COCO class names
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Annotate the image
    annotated_image = annotate_image(
        image,
        pred['boxes'].cpu().numpy(),
        pred['scores'].cpu().numpy(),
        pred['labels'].cpu().numpy(),
        COCO_CLASSES,
        confidence_threshold
    )
    
    return annotated_image

#Loaded a EfficientNetV3 model for species classification
modeld = models.efficientnet_b3(pretrained=True)

#Add a classifier layer at the end
num_ftrs = modeld.classifier[1].in_features
modeld.classifier[1] = nn.Linear(num_ftrs, 25)

#Loaded the weights obtained through training 
model_weights_path = './image/bird_image/best_model_FullTrainScratch.pth'
modeld.load_state_dict(torch.load(model_weights_path, map_location=device))

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

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image =image_path.convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0) 
    return image



def predict_image_class(image_path):
    image = preprocess_image(image_path)
    modeld.eval()
    with torch.no_grad():
        output = modeld(image)
        _, predicted = torch.max(output, 1)
        label = label_dict[predicted.item()]
        return label


def get_bbox_and_species(input_image_path):
    # Process the single input image
    annotated_image = detect_and_annotate_single_image(input_image_path, confidence_threshold=0.7)
    species = predict_image_class(input_image_path)
    return annotated_image, species

if __name__ == "__main__":
    # Example usage
    input_image_path = "/home/xintrean/Back-up/notes/sem_5/Project_1/YOLO/Birds_25/train/Sarus-Crane/Sarus-Crane_127.jpg"
    # annotated_image_, species = main(input_image_path)
    # print(species)
    # annotated_image_.show()  # Display the annotated image
    
