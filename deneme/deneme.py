import torch
import torchvision.models as models

model = models.resnet50(pretrained=True).eval()
x = torch.rand(1, 3, 224, 224)  # Örnek giriş
with torch.no_grad():
    features = model(x)
print(features.shape)
