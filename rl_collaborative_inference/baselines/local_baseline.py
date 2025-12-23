"""
Local baseline: Complete local inference without offloading
"""
import torch
import time
import numpy as np
from typing import Dict, Any


class LocalBaseline:
    """Local inference baseline - all computation on edge device"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()
    
    def inference(self, data, label):
        """
        Perform complete local inference
        :param data: Input data
        :param label: Ground truth label
        :return: accuracy, latency, info
        """
        data = data.to(self.device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            output = self.model(data)
        latency = time.time() - start_time
        
        # Calculate accuracy
        pred = torch.argmax(output, dim=1)
        # Ensure label is a tensor and on the same device
        if isinstance(label, (int, np.integer)):
            label_tensor = torch.tensor([label], device=pred.device, dtype=torch.long)
        elif isinstance(label, torch.Tensor):
            label_tensor = label.to(pred.device)
        else:
            label_tensor = torch.tensor([label], device=pred.device, dtype=torch.long)
        if label_tensor.dim() == 0:
            label_tensor = label_tensor.unsqueeze(0)
        if pred.shape != label_tensor.shape:
            if label_tensor.numel() == 1:
                label_tensor = label_tensor.expand_as(pred)
        accuracy = (pred == label_tensor).float().item()
        
        info = {
            'latency': latency,
            'accuracy': accuracy,
            'method': 'Local'
        }
        
        return accuracy, latency, info
    
    def evaluate(self, dataset, num_samples=None):
        """
        Evaluate on dataset
        :param dataset: Dataset to evaluate on
        :param num_samples: Number of samples to evaluate (None for all)
        :return: results dictionary
        """
        accuracies = []
        latencies = []
        
        num_samples = num_samples or len(dataset)
        
        for i in range(min(num_samples, len(dataset))):
            data, label = dataset[i]
            data = data.unsqueeze(0) if data.dim() == 3 else data
            
            accuracy, latency, _ = self.inference(data, label)
            accuracies.append(accuracy)
            latencies.append(latency)
        
        results = {
            'accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'method': 'Local'
        }
        
        return results

