# License: BSD
# Author: Sasank Chilamkurthy (modified by Heer Patel)

import os
import time
from tempfile import TemporaryDirectory
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

cudnn.benchmark = True
plt.ion()

# -----------------------
# Config
# -----------------------
DATA_DIR = '/Users/heerpatel/Desktop/fashion_project/data'
BATCH_SIZE = 4
NUM_WORKERS = 0
NUM_EPOCHS = 30     # bump to 30 for a smooth LR decay finish
USE_PRETRAINED = True

# -----------------------
# Helpers
# -----------------------
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_check(base=DATA_DIR):
    folders = ['train/couture', 'train/knockoffs', 'val/couture', 'val/knockoffs']
    for folder in folders:
        path = os.path.join(base, folder)
        try:
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        except FileNotFoundError:
            count = 0
        print(f"Number of images in {folder}: {count}")


def imshow(inp, title=None):
    if isinstance(inp, torch.Tensor):
        arr = inp.detach().cpu().numpy().transpose((1, 2, 0))
    else:
        arr = np.array(inp)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = np.clip(std * arr + mean, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# -----------------------
# Training Function
# -----------------------
def train_model(model, criterion, optimizer, scheduler,
                dataloaders, dataset_sizes, device,
                num_epochs=NUM_EPOCHS, start_epoch=0, checkpoint_path=None, best_acc=0.0):
    since = time.time()

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    best_model_wts = model.state_dict()

    with TemporaryDirectory() as tmp:
        best_path = os.path.join(tmp, "best_model_state.pt")
        torch.save(model.state_dict(), best_path)

        for epoch in range(start_epoch, num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                model.train(phase == 'train')
                running_loss, running_corrects = 0.0, 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train' and scheduler is not None:
                    scheduler.step()

                epoch_loss = running_loss / max(1, dataset_sizes[phase])
                epoch_acc = running_corrects.cpu().double() / max(1, dataset_sizes[phase])
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    train_loss_history.append(float(epoch_loss))
                    train_acc_history.append(float(epoch_acc))
                else:
                    val_loss_history.append(float(epoch_loss))
                    val_acc_history.append(float(epoch_acc))

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = model.state_dict()
                        torch.save(best_model_wts, best_path)

            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_acc': best_acc,
                }, checkpoint_path)
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        best_state = torch.load(best_path, map_location=device)
        model.load_state_dict(best_state)

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history


# -----------------------
# Visualization
# -----------------------
def visualize_model(model, dataloaders, class_names, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(8, 8))

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
    model.train(mode=was_training)


# -----------------------
# Logging
# -----------------------
def log_run(run_data, filename="training_log.json"):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                all_runs = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Warning: log file corrupted — resetting.")
            all_runs = []
    else:
        all_runs = []
    all_runs.append(run_data)
    with open(filename, "w") as f:
        json.dump(all_runs, f, indent=2)


# -----------------------
# Main
# -----------------------
def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=NUM_WORKERS)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = select_device()
    print(f"Using {device} device")
    count_check()

    # -----------------------
    # Load pretrained model
    # -----------------------
    weights_arg = models.ResNet18_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
    model_ft = models.resnet18(weights=weights_arg)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(0.3),  # slightly less dropout for fine-tuning
        nn.Linear(num_ftrs, len(class_names))
    )
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # -----------------------
    # Phase 1: Freeze backbone
    # -----------------------
    for param in model_ft.parameters():
        param.requires_grad = False
    for param in model_ft.fc.parameters():
        param.requires_grad = True

    print("🔒 Phase 1: Training classifier head only (backbone frozen)")

    optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)

    checkpoint_path = "fashion_classifier_checkpoint.pth"
    final_weights_path = "fashion_classifier_final.pth"

    # Train head for first 5 epochs
    model_ft, train_loss1, val_loss1, train_acc1, val_acc1 = train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler,
        dataloaders, dataset_sizes, device,
        num_epochs=5, checkpoint_path=checkpoint_path
    )

    # -----------------------
    # Phase 2: Unfreeze entire model
    # -----------------------
    print("\n🔓 Phase 2: Fine-tuning entire network with smaller LR")
    for param in model_ft.parameters():
        param.requires_grad = True

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=2.5e-5, momentum=0.9, weight_decay=1e-5)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

    model_ft, train_loss2, val_loss2, train_acc2, val_acc2 = train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler,
        dataloaders, dataset_sizes, device,
        num_epochs=25, checkpoint_path=checkpoint_path
    )

    # -----------------------
    # Combine results
    # -----------------------
    train_loss = train_loss1 + train_loss2
    val_loss = val_loss1 + val_loss2
    train_acc = train_acc1 + train_acc2
    val_acc = val_acc1 + val_acc2

    visualize_model(model_ft, dataloaders, class_names, device, num_images=6)

    torch.save(model_ft.state_dict(), final_weights_path)
    print(f"✅ Saved final model to {final_weights_path}")

    log_run({
        "model": "ResNet18 (two-phase fine-tune)",
        "epochs": len(train_acc),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "best_acc": float(max(val_acc)),
        "notes": "Phase1: frozen backbone (5 epochs, lr=1e-3); Phase2: full fine-tune (25 epochs, lr=2.5e-5, wd=1e-5)"
    })

    # Plot curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve (Two-Phase Fine-Tuning)')
    plt.tight_layout()
    plt.savefig("loss_curve_finetune.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve (Two-Phase Fine-Tuning)')
    plt.tight_layout()
    plt.savefig("acc_curve_finetune.png")
    plt.show()


if __name__ == "__main__":
    main()
