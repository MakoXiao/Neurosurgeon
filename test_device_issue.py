"""测试设备问题"""
import torch
from models.model_zoo import get_model

# 创建模型
model = get_model('resnet18', num_classes=101, pretrained=False)

# 加载权重
checkpoint = torch.load('checkpoints/resnet18/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"模型权重设备: {next(model.parameters()).device}")

# 移动到CPU
model = model.to('cpu')
print(f"移动到CPU后: {next(model.parameters()).device}")

# 测试推理
x = torch.randn(1, 3, 224, 224)
print(f"输入设备: {x.device}")

# 测试forward_to_split
output = model.forward_to_split(x, 1)
print(f"输出设备: {output.device}")
print(f"输出形状: {output.shape}")

print("\n✅ 测试成功!")


