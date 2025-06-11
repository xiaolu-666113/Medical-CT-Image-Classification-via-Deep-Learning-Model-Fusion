import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import num_classes

def get_model(name):
    if name == 'resnet':
        model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
    elif name == 'densenet':
        model = timm.create_model('densenet121', pretrained=True, num_classes=num_classes)
    elif name == 'efficientnet':
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    elif name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    elif name == 'convnext':
        model = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)
    elif name == 'regnet':
        model = timm.create_model('regnety_032', pretrained=True, num_classes=num_classes)
    elif name == 'swin':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
    elif name == 'mobilenetv3':
        model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=num_classes)
    elif name == 'beit':
        model = timm.create_model('beit_base_patch16_224', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型：{name}")
    return model

class FusionModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.ones(len(models)))

    def forward(self, x):
        logits = [m(x) for m in self.models]
        logits_stack = torch.stack(logits, dim=0)  # (num_models, batch_size, num_classes)
        weights = F.softmax(self.weights, dim=0).view(-1, 1, 1)
        return (weights * logits_stack).sum(dim=0)

    def get_normalized_weights(self):
        with torch.no_grad():
            weights = torch.softmax(self.weights, dim=0)
            return weights.cpu().numpy()

