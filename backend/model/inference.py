# backend/model/inference.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 🔹 Load your model architecture
def load_model(weights_path="backend/model/fashion_classifier_final.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a ResNet18 model
    model = models.resnet18(pretrained=False)

    # Modify the final layer to match your number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # assuming 2 classes: couture vs knockoff

    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# 🔹 Define a preprocessing pipeline (same as your training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])


# 🔹 Define prediction helper
def predict(model, image_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)

    label = "Couture" if preds.item() == 0 else "Knockoff"
    return label

def create_model(num_classes=2):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Identity(),  # placeholder for index 0 (if existed before)
        nn.Linear(num_ftrs, num_classes)
    )
    return model



def load_model(device):
    model = create_model(num_classes=2)
    base_dir = os.path.dirname(__file__)  # directory where this file (inference.py) lives
    weights_path = os.path.join(base_dir, "fashion_classifier_final.pth")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

