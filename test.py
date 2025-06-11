import torch
import torch.nn as nn
from config import *
from data_loader import get_dataloaders
from models import get_model, FusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, val_loader = get_dataloaders()

base_models = [get_model(name).to(device) for name in model_names]
model = FusionModel(base_models).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

correct = 0
total = 0
total_loss = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

avg_loss = total_loss / total
accuracy = correct / total

print(f"\n🧪 测试集结果：\n准确率: {accuracy:.4f} | 平均损失: {avg_loss:.4f}")
