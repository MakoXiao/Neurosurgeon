"""
调试准确率问题
"""
import os
import sys
import torch
import numpy as np

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up to Neurosurgeon directory
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from models.AlexNet import AlexNet
from src.dataset_loader import get_caltech101_dataloader
from src.model_partition import ModelPartitioner
from src.pruning import PruningManager

# Load model and dataset
print("Loading model and dataset...")
model = AlexNet(input_channels=3, num_classes=101)
model.eval()
_, test_dataset = get_caltech101_dataloader('../data/caltech-101', batch_size=1, split='test')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Test full model accuracy
print("\n=== Testing Full Model ===")
correct_full = 0
total = 0
for i in range(min(20, len(test_dataset))):
    img, label = test_dataset[i]
    img = img.unsqueeze(0).to(device)
    model = model.to(device)
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
    if pred == label:
        correct_full += 1
    total += 1
    if i < 5:
        print(f"Sample {i}: Label={label}, Pred={pred}, Match={pred==label}, Output shape={output.shape}")

print(f"Full model accuracy on {total} samples: {correct_full/total:.4f}")

# Test partitioned model
print("\n=== Testing Partitioned Model ===")
partitioner = ModelPartitioner(model)
pruning_manager = PruningManager(pruning_type='structured')

partition_point = 4
compression_rate = 0.5

edge_model, cloud_model = partitioner.partition(partition_point)
edge_model = edge_model.to(device)
cloud_model = cloud_model.to(device)

correct_partitioned = 0
total = 0

for i in range(min(20, len(test_dataset))):
    img, label = test_dataset[i]
    img = img.unsqueeze(0).to(device)
    
    # Edge inference
    with torch.no_grad():
        edge_output = edge_model(img)
    
    # Prune
    pruned_feature, pruning_info = pruning_manager.compress(edge_output, compression_rate)
    
    # Decompress
    recovered_feature = pruning_manager.decompress(pruned_feature, pruning_info, device)
    
    # Cloud inference
    with torch.no_grad():
        cloud_output = cloud_model(recovered_feature)
    
    pred = torch.argmax(cloud_output, dim=1)
    
    # Check label type and comparison
    if isinstance(label, (int, np.integer)):
        label_tensor = torch.tensor([label], device=device, dtype=torch.long)
    else:
        label_tensor = label.to(device) if hasattr(label, 'to') else torch.tensor([label], device=device, dtype=torch.long)
    
    if label_tensor.dim() == 0:
        label_tensor = label_tensor.unsqueeze(0)
    
    if pred.shape != label_tensor.shape:
        if label_tensor.numel() == 1:
            label_tensor = label_tensor.expand_as(pred)
    
    accuracy = (pred == label_tensor).float().item()
    
    if accuracy > 0:
        correct_partitioned += 1
    
    total += 1
    
    if i < 5:
        print(f"Sample {i}: Label={label} (type={type(label)}), Pred={pred.item()}, "
              f"Label_tensor={label_tensor.item()}, Accuracy={accuracy}, "
              f"Match={pred.item() == label_tensor.item()}")
        print(f"  Cloud output shape: {cloud_output.shape}, Max: {cloud_output.max().item():.2f}, "
              f"Min: {cloud_output.min().item():.2f}")

print(f"\nPartitioned model accuracy on {total} samples: {correct_partitioned/total:.4f}")

# Check if model outputs are valid
print("\n=== Checking Model Outputs ===")
img, label = test_dataset[0]
img = img.unsqueeze(0).to(device)
with torch.no_grad():
    output = model(img)
    print(f"Output shape: {output.shape}")
    print(f"Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, "
          f"mean={output.mean().item():.4f}, std={output.std().item():.4f}")
    print(f"Top 5 predictions: {torch.topk(output, 5, dim=1)[1].squeeze().tolist()}")
    print(f"True label: {label}")

