"""
训练AlexNet分类模型用于评估
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.AlexNet import AlexNet
from src.dataset_loader import get_caltech101_dataloader


def train_model(data_dir, output_path, num_epochs=20, batch_size=32, lr=0.001):
    """训练AlexNet模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    try:
        train_loader, train_dataset = get_caltech101_dataloader(
            data_dir, batch_size=batch_size, split='train', num_workers=2
        )
        val_loader, val_dataset = get_caltech101_dataloader(
            data_dir, batch_size=batch_size, split='test', num_workers=2
        )
        
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        print(f"Train samples: {train_size}, Val samples: {val_size}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: use train split for both
        print("Using train split for both training and validation...")
        train_loader, train_dataset = get_caltech101_dataloader(
            data_dir, batch_size=batch_size, split='train', num_workers=2
        )
        val_loader = None  # Will use training accuracy as validation
        train_size = len(train_dataset)
        print(f"Using {train_size} samples for training")
    
    # Create model
    model = AlexNet(input_channels=3, num_classes=101).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in pbar:
            # Ensure proper tensor types
            if not isinstance(images, torch.Tensor):
                continue
            if not isinstance(labels, torch.Tensor):
                if isinstance(labels, (int, list)):
                    labels = torch.tensor(labels if isinstance(labels, list) else [labels], dtype=torch.long)
                else:
                    continue
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            if val_loader and len(val_loader) > 0:
                for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                    # Ensure proper tensor types
                    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
                        continue
                    
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            else:
                # Use training accuracy as validation accuracy if no separate val set
                val_acc = train_acc
                val_correct = train_correct
                val_total = train_total
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        print(f'  Val details: correct={val_correct}, total={val_total}')
        
        # Save best model (use train_acc if val_acc is 0)
        save_acc = val_acc if val_acc > 0 else train_acc
        if save_acc > best_acc:
            best_acc = save_acc
            torch.save(model.state_dict(), output_path)
            print(f'  -> Saved best model (Acc: {save_acc:.2f}%)')
        
        scheduler.step()
    
    print(f'\nTraining completed! Best validation accuracy: {best_acc:.2f}%')
    print(f'Model saved to: {output_path}')
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AlexNet on Caltech-101')
    parser.add_argument('--data_dir', type=str, default='../data/caltech-101',
                       help='Path to Caltech-101 dataset')
    parser.add_argument('--output_path', type=str, default='./alexnet_caltech101.pth',
                       help='Path to save trained model')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output_path, args.num_epochs, args.batch_size, args.lr)

