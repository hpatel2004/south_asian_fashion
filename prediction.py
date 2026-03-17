import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# same device logic as your training script
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path):
    device = select_device()

    # rebuild EXACT same architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 2)  # couture vs knockoffs
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, device


def predict(image_path, model, device):
    # MUST match your val transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)

    classes = ["couture", "knockoffs"]

    print(f"Prediction: {classes[pred.item()]}")
    print(f"Confidence: {probs[0][pred.item()]:.4f}")


if __name__ == "__main__":
    model_path = "backend/model/fashion_classifier_final.pth"
    image_path = sys.argv[1]

    model, device = load_model(model_path)
    predict(image_path, model, device)