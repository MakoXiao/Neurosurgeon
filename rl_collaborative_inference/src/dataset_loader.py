"""
Dataset loader for Caltech-101
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class Caltech101Dataset(Dataset):
    """Caltech-101 dataset loader"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        :param root_dir: root directory of Caltech-101
        :param split: 'train' or 'test'
        :param transform: image transform
        """
        self.root_dir = os.path.join(root_dir, '101_ObjectCategories')
        self.split = split
        self.transform = transform
        
        # Get all categories
        categories = sorted(os.listdir(self.root_dir))
        if 'BACKGROUND_Google' in categories:
            categories.remove('BACKGROUND_Google')
        
        self.categories = categories
        self.class_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        
        # Load images
        self.images = []
        self.labels = []
        
        for category in categories:
            category_dir = os.path.join(self.root_dir, category)
            if not os.path.isdir(category_dir):
                continue
            
            images = [f for f in os.listdir(category_dir) if f.endswith('.jpg')]
            
            # Simple train/test split (80/20)
            split_idx = int(len(images) * 0.8)
            if split == 'train':
                split_images = images[:split_idx]
            else:
                split_images = images[split_idx:]
            
            for img_name in split_images:
                img_path = os.path.join(category_dir, img_name)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[category])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_caltech101_dataloader(root_dir, batch_size=1, split='train', num_workers=0):
    """
    Get Caltech-101 dataloader
    :param root_dir: root directory
    :param batch_size: batch size
    :param split: 'train' or 'test'
    :param num_workers: number of workers
    :return: dataloader
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = Caltech101Dataset(root_dir, split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                          num_workers=num_workers)
    
    return dataloader, dataset

