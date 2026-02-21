"""
Training script for deep learning classification models
Trains AlexNet, VGG-11, ResNet-18, and MobileNet-V2 on Caltech-101 dataset
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.dataset_loader import get_caltech101_dataloader
from models.AlexNet import AlexNet
from models.VggNet import VGG
from models.MobileNet import MobileNet
import torchvision.models as torchvision_models


def create_vgg11(input_channels=3, num_classes=101):
    """
    Create VGG-11 model
    Note: Using VGG-16 architecture from models.VggNet
    """
    from models.VggNet import vgg16
    model = vgg16(input_channels=input_channels, num_classes=num_classes)
    return model


def create_resnet18(input_channels=3, num_classes=101):
    """
    Create ResNet-18 model from torchvision
    """
    model = torchvision_models.resnet18(pretrained=False)
    # Modify the final layer for Caltech-101 (101 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(model, train_loader, val_loader, args, model_name):
    """
    Train a classification model

    Args:
        model: The neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Training arguments
        model_name: Name of the model (for saving)

    Returns:
        best_model_state: State dict of the best model
        training_history: Dict containing training metrics
    """
    device = args.device
    model = model.to(device)

    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_acc:.2f}%'
            })

        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_acc = 100 * val_correct / val_total if val_total > 0 else 0
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{val_acc:.2f}%'
                })

        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save training history
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['lr'].append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    print(f"\n{'='*60}")
    print(f"Training completed for {model_name}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}\n")

    return best_model_state, training_history


def main():
    parser = argparse.ArgumentParser(description='Train classification models on Caltech-101')

    # Model selection
    parser.add_argument('--model', type=str, default='all',
                       choices=['alexnet', 'vgg11', 'resnet18', 'mobilenetv2', 'all'],
                       help='Model to train (default: all)')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../data/caltech-101',
                       help='Path to Caltech-101 dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--step_size', type=int, default=10,
                       help='Period of learning rate decay (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Multiplicative factor of learning rate decay (default: 0.1)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience (default: 5)')

    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training (default: cuda)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                       help='Directory to save trained models (default: ./trained_models)')

    args = parser.parse_args()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define models to train
    if args.model == 'all':
        models_to_train = {
            'alexnet': {'model_class': AlexNet, 'batch_size': 32, 'lr': 0.001},
            'vgg11': {'model_class': create_vgg11, 'batch_size': 16, 'lr': 0.0005},
            'resnet18': {'model_class': create_resnet18, 'batch_size': 32, 'lr': 0.001},
            'mobilenetv2': {'model_class': MobileNet, 'batch_size': 32, 'lr': 0.001}
        }
    else:
        model_classes = {
            'alexnet': AlexNet,
            'vgg11': create_vgg11,
            'resnet18': create_resnet18,
            'mobilenetv2': MobileNet
        }
        models_to_train = {
            args.model: {
                'model_class': model_classes[args.model],
                'batch_size': args.batch_size,
                'lr': args.lr
            }
        }

    # Training summary
    training_summary = {}

    # Train each model
    for model_name, config in models_to_train.items():
        print(f"\n{'='*70}")
        print(f"Starting training for: {model_name.upper()}")
        print(f"{'='*70}")

        # Update batch size and learning rate for specific model
        args.batch_size = config['batch_size']
        args.lr = config['lr']

        # Load dataset
        print("\nLoading Caltech-101 dataset...")
        train_loader, train_dataset = get_caltech101_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            split='train',
            num_workers=args.num_workers
        )
        val_loader, val_dataset = get_caltech101_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            split='test',
            num_workers=args.num_workers
        )

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        if len(train_dataset) == 0:
            print(f"Warning: No training data found, skipping {model_name}")
            continue

        # Create model
        print(f"\nCreating {model_name} model...")
        model = config['model_class'](input_channels=3, num_classes=101)

        # Train model
        best_model_state, training_history = train_model(
            model, train_loader, val_loader, args, model_name
        )

        # Save model
        model_save_path = os.path.join(args.output_dir, f'{model_name}_caltech101.pth')
        torch.save(best_model_state, model_save_path)
        print(f"Model saved to: {model_save_path}")

        # Save training history
        history_save_path = os.path.join(args.output_dir, f'{model_name}_training_history.json')
        with open(history_save_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"Training history saved to: {history_save_path}")

        # Add to summary
        training_summary[model_name] = {
            'best_val_acc': max(training_history['val_acc']),
            'final_train_acc': training_history['train_acc'][-1],
            'model_path': model_save_path,
            'training_epochs': len(training_history['train_acc'])
        }

    # Save overall summary
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)

    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    for model_name, summary in training_summary.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Best Val Accuracy: {summary['best_val_acc']:.2f}%")
        print(f"  Final Train Accuracy: {summary['final_train_acc']:.2f}%")
        print(f"  Training Epochs: {summary['training_epochs']}")
        print(f"  Model Path: {summary['model_path']}")
    print(f"\n{'='*70}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
