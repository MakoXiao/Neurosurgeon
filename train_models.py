"""
模型训练脚本
训练ResNet18、VGG11、MobileNetV2、AlexNet用于Caltech-101分类
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import time
from tqdm import tqdm
import json

from dataset.caltech101_loader import get_caltech101_dataloaders
from models.model_zoo import get_model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def train_model(model_name, data_dir, save_dir, num_epochs=50, 
                batch_size=32, lr=0.001, device='cuda'):
    """
    训练模型
    
    Args:
        model_name: 模型名称
        data_dir: 数据目录
        save_dir: 保存目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        device: 设备
    """
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"{'='*60}\n")
    
    # 创建保存目录
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    train_loader, test_loader, num_classes = get_caltech101_dataloaders(
        data_dir, batch_size=batch_size, num_workers=4
    )
    
    # 创建模型
    print(f"创建模型 {model_name}...")
    model = get_model(model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'best_acc': 0.0,
        'best_epoch': 0
    }
    
    best_acc = 0.0
    
    # 训练循环
    print(f"\n开始训练...")
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 打印结果
        print(f"\n结果:")
        print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"  测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            history['best_acc'] = best_acc
            history['best_epoch'] = epoch
            
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, best_model_path)
            print(f"  保存最佳模型 (准确率: {best_acc:.2f}%)")
        
        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, checkpoint_path)
    
    # 保存训练历史
    history_path = os.path.join(model_save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"最佳准确率: {best_acc:.2f}% (Epoch {history['best_epoch']})")
    print(f"模型保存在: {model_save_dir}")
    print(f"{'='*60}\n")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='训练分类模型')
    parser.add_argument('--model', type=str, default='all',
                       choices=['resnet18', 'vgg11', 'mobilenetv2', 'alexnet', 'all'],
                       help='模型名称')
    parser.add_argument('--data_dir', type=str, 
                       default='/opt/03-ai/01-proj/Neurosurgeon/data/caltech-101',
                       help='数据目录')
    parser.add_argument('--save_dir', type=str, 
                       default='/opt/03-ai/01-proj/Neurosurgeon/checkpoints',
                       help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备')
    
    args = parser.parse_args()
    
    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    print(f"\n训练配置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  保存目录: {args.save_dir}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  设备: {args.device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练模型
    if args.model == 'all':
        models = ['resnet18', 'vgg11', 'mobilenetv2', 'alexnet']
    else:
        models = [args.model]
    
    all_histories = {}
    
    for model_name in models:
        start_time = time.time()
        
        history = train_model(
            model_name=model_name,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device
        )
        
        elapsed_time = time.time() - start_time
        print(f"{model_name} 训练耗时: {elapsed_time/60:.2f} 分钟\n")
        
        all_histories[model_name] = history
    
    # 保存所有模型的训练历史
    summary_path = os.path.join(args.save_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_histories, f, indent=4)
    
    # 打印总结
    print("\n" + "="*60)
    print("训练总结")
    print("="*60)
    for model_name, history in all_histories.items():
        print(f"{model_name:15s}: 最佳准确率 {history['best_acc']:.2f}% (Epoch {history['best_epoch']})")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

