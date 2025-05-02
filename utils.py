import torch
import torch.nn as nn

def create_model():
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, 5, padding=0), nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(400, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10)
    )
    return model

def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0.0; correct = 0; total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(); outputs = model(images); loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0); correct += (predicted == labels).sum().item()
    return train_loss / total, 100. * correct / total

def validate_epoch(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images); loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0); correct += (predicted == labels).sum().item()
    return val_loss / total, 100. * correct / total