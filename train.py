import torch
import torch.nn as nn
from config import *
from data_loader import get_dataloaders
from models import get_model, FusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备：{device}")

train_loader, val_loader = get_dataloaders()
print("📦 数据加载完成！")

# 初始化多个基础模型
base_models = [get_model(name).to(device) for name in model_names]
model = FusionModel(base_models).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            outputs = model(x)
            loss = criterion(outputs, y)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

best_val_acc = 0
patience = 5
patience_counter = 0
save_path = 'best_model.pth'

for epoch in range(num_epochs):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # ✅ 每轮显示融合权重
    norm_weights = model.get_normalized_weights()  # 确保 FusionModel 中有该方法
    print("📊 模型融合权重分布：")
    for name, w in zip(model_names, norm_weights):
        print(f"  {name}: {w:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"✅ 新最佳模型已保存！Val Acc: {val_acc:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"⚠️ 验证准确率未提升（{patience_counter}/{patience}）")

    if patience_counter >= patience:
        print(f"🛑 Early stopping 触发，验证准确率连续 {patience} 轮无提升")
        break
