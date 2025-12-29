"""
Caltech-101数据集加载器
支持训练集和测试集的加载、数据增强等
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class Caltech101Dataset(Dataset):
    """Caltech-101数据集类"""
    
    def __init__(self, root_dir, transform=None, train=True, train_ratio=0.8):
        """
        Args:
            root_dir: 数据集根目录，应包含101_ObjectCategories文件夹
            transform: 数据转换
            train: 是否为训练集
            train_ratio: 训练集比例
        """
        self.root_dir = os.path.join(root_dir, '101_ObjectCategories')
        self.transform = transform
        self.train = train
        self.train_ratio = train_ratio
        
        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(self.root_dir) 
                              if os.path.isdir(os.path.join(self.root_dir, d)) 
                              and d != 'BACKGROUND_Google'])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 加载所有图像路径和标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # 获取该类别的所有图像
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))
        
        # 随机打乱并划分训练集和测试集
        random.seed(42)
        random.shuffle(self.samples)
        
        split_idx = int(len(self.samples) * self.train_ratio)
        if self.train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个黑色图像
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_caltech101_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    获取Caltech-101数据加载器
    
    Args:
        data_dir: 数据集目录
        batch_size: 批次大小
        num_workers: 工作进程数
    
    Returns:
        train_loader, test_loader, num_classes
    """
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 测试集数据转换
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = Caltech101Dataset(
        root_dir=data_dir,
        transform=train_transform,
        train=True,
        train_ratio=0.8
    )
    
    test_dataset = Caltech101Dataset(
        root_dir=data_dir,
        transform=test_transform,
        train=False,
        train_ratio=0.8
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = len(train_dataset.classes)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别数量: {num_classes}")
    
    return train_loader, test_loader, num_classes


if __name__ == '__main__':
    # 测试数据加载器
    data_dir = '/opt/03-ai/01-proj/Neurosurgeon/data/caltech-101'
    train_loader, test_loader, num_classes = get_caltech101_dataloaders(
        data_dir, batch_size=32, num_workers=2
    )
    
    print(f"\n类别数量: {num_classes}")
    
    # 测试加载一个批次
    for images, labels in train_loader:
        print(f"图像批次形状: {images.shape}")
        print(f"标签批次形状: {labels.shape}")
        print(f"标签范围: {labels.min()} - {labels.max()}")
        break

