"""
测试训练好的模型准确率
"""
import os
import sys
import torch

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from models.AlexNet import AlexNet
from rl_collaborative_inference.src.dataset_loader import get_caltech101_dataloader

# 加载训练好的模型
model = AlexNet(3, 101)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'alexnet_caltech101.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Loaded model from {model_path}")
else:
    print(f"Model not found at {model_path}")
    sys.exit(1)

model.eval()

# 加载测试集
test_loader, test_dataset = get_caltech101_dataloader('../data/caltech-101', batch_size=1, split='test')
print(f"Test dataset size: {len(test_dataset)}")

# 测试准确率
correct = 0
total = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

for i, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        
        # 打印前5个样本的详细信息
        if i < 5:
            print(f"Sample {i}: pred={pred.item()}, label={labels.item()}, match={pred.item()==labels.item()}")
            print(f"  Output shape: {outputs.shape}, Output max: {outputs.max().item():.2f}, Output min: {outputs.min().item():.2f}")
        
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    if i >= 100:  # 只测试前100个样本
        break

print(f"\nTest accuracy on {total} samples: {100*correct/total:.2f}%")
print(f"Correct: {correct}, Total: {total}")

