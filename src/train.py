import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# CONFIG
# =========================

BASE_PATH = "/content/drive/MyDrive/dataset"  # Change this path if needed
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATA TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# =========================
# LOAD DATA
# =========================

train_data = datasets.ImageFolder(os.path.join(BASE_PATH, "train"), transform=train_transform)
val_data = datasets.ImageFolder(os.path.join(BASE_PATH, "val"), transform=eval_transform)
test_data = datasets.ImageFolder(os.path.join(BASE_PATH, "test"), transform=eval_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_data.classes

print("Classes:", class_names)
print("Train images:", len(train_data))
print("Validation images:", len(val_data))
print("Test images:", len(test_data))


# =========================
# MODEL
# =========================

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# =========================
# TRAINING
# =========================

best_model_weights = copy.deepcopy(model.state_dict())
best_val_accuracy = 0.0

train_losses = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_data)
    train_losses.append(epoch_loss)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)

    print(f"Training loss: {epoch_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_weights = copy.deepcopy(model.state_dict())


# =========================
# SAVE BEST MODEL
# =========================

model.load_state_dict(best_model_weights)

os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/best_model.pth")

print("\nBest validation accuracy:", best_val_accuracy)


# =========================
# TEST EVALUATION
# =========================

model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

report = classification_report(all_labels, all_predictions, target_names=class_names)
matrix = confusion_matrix(all_labels, all_predictions)

print("\nClassification report:")
print(report)

print("\nConfusion matrix:")
print(matrix)

with open("results/metrics.txt", "w") as f:
    f.write("Osteoporosis Detection Results\n")
    f.write("==============================\n\n")
    f.write(f"Best validation accuracy: {best_val_accuracy:.4f}\n\n")
    f.write("Classification report:\n")
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(str(matrix))


# =========================
# PLOT TRAINING CURVES
# =========================

plt.figure()
plt.plot(train_losses, label="Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig("results/training_loss.png")
plt.close()

plt.figure()
plt.plot(val_accuracies, label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.savefig("results/validation_accuracy.png")
plt.close()
