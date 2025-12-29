"""压缩模块"""
from .pruning_compression import PruningCompressor, StructuredPruningCompressor, UnstructuredPruningCompressor, AdaptivePruningCompressor, compute_compression_ratio
__all__ = ['PruningCompressor', 'StructuredPruningCompressor', 'UnstructuredPruningCompressor', 'AdaptivePruningCompressor', 'compute_compression_ratio']
