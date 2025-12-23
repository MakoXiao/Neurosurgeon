"""
JALAD baseline: Joint Adaptive Learning and DNN Partitioning
Based on autoencoder-based feature compression
"""
import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.model_partition import ModelPartitioner
from src.pruning import PruningManager


class AutoencoderCompressor(nn.Module):
    """Lightweight autoencoder for feature compression (JALAD style)"""
    
    def __init__(self, input_dim, compression_ratio=0.5, max_dim=100000):
        super().__init__()
        # Limit input dimension to avoid memory issues
        self.max_dim = min(input_dim, max_dim)
        self.compression_ratio = compression_ratio
        
        # Simple autoencoder architecture
        hidden_dim = int(self.max_dim * compression_ratio)
        compressed_dim = hidden_dim // 2
        
        self.encoder = nn.Sequential(
            nn.Linear(self.max_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, compressed_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.max_dim)
        )
    
    def encode(self, x):
        """Encode features"""
        # Flatten if needed
        if x.dim() > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        
        # Pad or crop to max_dim
        if x.shape[1] > self.max_dim:
            x = x[:, :self.max_dim]  # Crop
        elif x.shape[1] < self.max_dim:
            padding = torch.zeros(x.shape[0], self.max_dim - x.shape[1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)  # Pad
        
        return self.encoder(x)
    
    def decode(self, encoded):
        """Decode features"""
        return self.decoder(encoded)
    
    def forward(self, x):
        """Forward pass"""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


class JALADBaseline:
    """JALAD baseline with autoencoder compression"""
    
    def __init__(self, model, dataset, edge_device='cpu', cloud_device='cpu',
                 network_bandwidth=10.0, compression_ratio=0.5, partition_point=4):
        self.model = model
        self.dataset = dataset
        self.edge_device = edge_device
        self.cloud_device = cloud_device
        self.network_bandwidth = network_bandwidth
        self.compression_ratio = compression_ratio
        self.partition_point = partition_point
        
        # Partition model
        self.partitioner = ModelPartitioner(model)
        edge_model, cloud_model = self.partitioner.partition(partition_point)
        self.edge_model = edge_model.to(edge_device)
        self.cloud_model = cloud_model.to(cloud_device)
        
        # Create autoencoder
        # Estimate feature dimension (simplified)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(edge_device)
            edge_output = self.edge_model(dummy_input)
            feature_dim = edge_output.numel()
        
        # Use fixed max dimension for autoencoder to avoid memory issues
        max_feature_dim = 50000  # Limit to 50K features for autoencoder
        self.autoencoder = AutoencoderCompressor(
            input_dim=feature_dim, 
            compression_ratio=compression_ratio,
            max_dim=max_feature_dim
        ).to('cpu')  # Use CPU for autoencoder to save GPU memory
        self.autoencoder.eval()
    
    def inference(self, data, label):
        """
        Perform JALAD-style inference
        :param data: Input data
        :param label: Ground truth label
        :return: accuracy, latency, info
        """
        data = data.to(self.edge_device)
        total_start_time = time.time()
        
        # Edge inference
        edge_start_time = time.time()
        with torch.no_grad():
            edge_output = self.edge_model(data)
        edge_time = time.time() - edge_start_time
        
        # Compress with autoencoder
        compress_start_time = time.time()
        with torch.no_grad():
            # Flatten feature
            batch_size = edge_output.size(0)
            original_shape = edge_output.shape
            feature_flat = edge_output.view(batch_size, -1)
            original_feature_dim = feature_flat.shape[1]
            
            # Store original shape for recovery
            self._last_original_shape = original_shape
            self._last_original_feature_dim = original_feature_dim
            
            # Move to CPU for encoding (autoencoder handles dimension adjustment)
            feature_flat_cpu = feature_flat.cpu()
            compressed = self.autoencoder.encode(feature_flat_cpu)
        compress_time = time.time() - compress_start_time
        
        # Calculate transmission time
        compressed_size_bytes = compressed.numel() * 4  # float32
        compressed_size_mb = compressed_size_bytes / (1024 * 1024)
        transmission_time = compressed_size_mb / self.network_bandwidth
        
        # Decompress and cloud inference
        cloud_start_time = time.time()
        with torch.no_grad():
            # Decompress on CPU
            decompressed_cpu = self.autoencoder.decode(compressed)
            # Move to cloud device
            decompressed_flat = decompressed_cpu.to(self.cloud_device)
            
            # Recover original feature shape
            original_shape = getattr(self, '_last_original_shape', None)
            original_feature_dim = getattr(self, '_last_original_feature_dim', None)
            
            if original_shape is None or original_feature_dim is None:
                # Fallback: try to infer from edge_output
                original_shape = edge_output.shape
                original_feature_dim = edge_output.numel() // batch_size
            else:
                # Ensure we have the correct original feature dimension
                if original_feature_dim != edge_output.numel() // batch_size:
                    original_feature_dim = edge_output.numel() // batch_size
                    original_shape = edge_output.shape
            
            # Adjust decompressed to match original dimension
            decompressed_dim = decompressed_flat.shape[1]
            
            if decompressed_dim < original_feature_dim:
                # Pad to original size
                padding = torch.zeros(batch_size, original_feature_dim - decompressed_dim, 
                                    device=self.cloud_device, dtype=decompressed_flat.dtype)
                decompressed_flat = torch.cat([decompressed_flat, padding], dim=1)
            elif decompressed_dim > original_feature_dim:
                # Crop to original size
                decompressed_flat = decompressed_flat[:, :original_feature_dim]
            
            # Reshape to original feature shape
            try:
                decompressed = decompressed_flat.view(original_shape)
            except RuntimeError as e:
                # Fallback: reshape to match edge_output shape
                print(f"Warning: Cannot reshape to {original_shape}, using edge_output shape: {e}")
                decompressed = decompressed_flat.view(edge_output.shape)
            
            # Cloud inference
            # Ensure decompressed shape matches what cloud_model expects
            if decompressed.shape != edge_output.shape:
                # Try to reshape or pad/crop to match
                try:
                    # If dimensions don't match, try to reshape
                    if decompressed.numel() == edge_output.numel():
                        decompressed = decompressed.view(edge_output.shape)
                    else:
                        # Pad or crop to match
                        decompressed_flat = decompressed.view(1, -1)
                        edge_flat = edge_output.view(1, -1)
                        if decompressed_flat.shape[1] < edge_flat.shape[1]:
                            padding = torch.zeros(1, edge_flat.shape[1] - decompressed_flat.shape[1], 
                                                device=self.cloud_device, dtype=decompressed_flat.dtype)
                            decompressed_flat = torch.cat([decompressed_flat, padding], dim=1)
                        elif decompressed_flat.shape[1] > edge_flat.shape[1]:
                            decompressed_flat = decompressed_flat[:, :edge_flat.shape[1]]
                        decompressed = decompressed_flat.view(edge_output.shape)
                except Exception as e:
                    print(f"Warning: Cannot reshape decompressed feature: {e}")
                    # Use edge_output directly as fallback (no compression effect)
                    decompressed = edge_output
            
            cloud_output = self.cloud_model(decompressed)
        cloud_time = time.time() - cloud_start_time
        
        total_latency = time.time() - total_start_time
        
        # Calculate accuracy
        pred = torch.argmax(cloud_output, dim=1)
        # Ensure label is a tensor and on the correct device
        if isinstance(label, (int, np.integer)):
            label_tensor = torch.tensor([label], device=self.cloud_device, dtype=torch.long)
        elif isinstance(label, torch.Tensor):
            label_tensor = label.to(self.cloud_device)
            if label_tensor.dim() == 0:
                label_tensor = label_tensor.unsqueeze(0)
        else:
            label_tensor = torch.tensor([label], device=self.cloud_device, dtype=torch.long)
        
        # Ensure same shape
        if pred.shape != label_tensor.shape:
            if label_tensor.numel() == 1:
                label_tensor = label_tensor.expand_as(pred)
        
        accuracy = (pred == label_tensor).float().item()
        
        # Calculate compression ratio
        original_size = edge_output.numel() * 4
        compressed_size = compressed.numel() * 4
        compression_rate = original_size / compressed_size if compressed_size > 0 else 1.0
        
        info = {
            'latency': total_latency,
            'edge_time': edge_time,
            'transmission_time': transmission_time,
            'cloud_time': cloud_time,
            'accuracy': accuracy,
            'compression_rate': compression_rate,
            'method': 'JALAD'
        }
        
        return accuracy, total_latency, info
    
    def evaluate(self, num_samples=50):
        """
        Evaluate on dataset
        :param num_samples: Number of samples to evaluate
        :return: results dictionary
        """
        accuracies = []
        latencies = []
        compression_rates = []
        
        num_samples = min(num_samples, len(self.dataset))
        
        for i in range(num_samples):
            data, label = self.dataset[i]
            data = data.unsqueeze(0) if data.dim() == 3 else data
            
            accuracy, latency, info = self.inference(data, label)
            accuracies.append(accuracy)
            latencies.append(latency)
            compression_rates.append(info['compression_rate'])
        
        results = {
            'accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'compression_rate': np.mean(compression_rates),
            'method': 'JALAD'
        }
        
        return results

