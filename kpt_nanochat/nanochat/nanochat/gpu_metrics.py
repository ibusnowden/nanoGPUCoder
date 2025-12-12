"""
GPU metrics collection for training and inference monitoring.
Collects GPU utilization, memory usage, and performance metrics.
"""

import torch
import time
from typing import Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class GPUMetrics:
    """Container for GPU performance metrics"""
    timestamp: float
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    gpu_memory_max_allocated_mb: float
    gpu_utilization_percent: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class GPUMetricsCollector:
    """Collects GPU metrics during training and inference"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.metrics_history = []
        
        # Try to import pynvml for GPU utilization monitoring
        self.nvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_available = True
            self.pynvml = pynvml
            # Get handle for the current GPU
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        except (ImportError, Exception):
            # pynvml not available, will only collect memory metrics
            pass
    
    def collect(self) -> GPUMetrics:
        """Collect current GPU metrics"""
        if not torch.cuda.is_available():
            return GPUMetrics(
                timestamp=time.time(),
                gpu_memory_allocated_mb=0.0,
                gpu_memory_reserved_mb=0.0,
                gpu_memory_max_allocated_mb=0.0,
            )
        
        # Memory metrics (always available with PyTorch)
        memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        memory_max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        # GPU utilization (requires pynvml)
        gpu_util = None
        if self.nvml_available:
            try:
                util_info = self.pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                gpu_util = float(util_info.gpu)
            except Exception:
                pass
        
        metrics = GPUMetrics(
            timestamp=time.time(),
            gpu_memory_allocated_mb=memory_allocated,
            gpu_memory_reserved_mb=memory_reserved,
            gpu_memory_max_allocated_mb=memory_max_allocated,
            gpu_utilization_percent=gpu_util,
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def get_summary(self) -> Dict:
        """Get summary statistics of collected metrics"""
        if not self.metrics_history:
            return {}
        
        memory_allocated = [m.gpu_memory_allocated_mb for m in self.metrics_history]
        memory_reserved = [m.gpu_memory_reserved_mb for m in self.metrics_history]
        
        summary = {
            "num_samples": len(self.metrics_history),
            "memory_allocated_avg_mb": sum(memory_allocated) / len(memory_allocated),
            "memory_allocated_max_mb": max(memory_allocated),
            "memory_reserved_avg_mb": sum(memory_reserved) / len(memory_reserved),
            "memory_reserved_max_mb": max(memory_reserved),
        }
        
        # Add GPU utilization if available
        gpu_utils = [m.gpu_utilization_percent for m in self.metrics_history if m.gpu_utilization_percent is not None]
        if gpu_utils:
            summary["gpu_utilization_avg_percent"] = sum(gpu_utils) / len(gpu_utils)
            summary["gpu_utilization_max_percent"] = max(gpu_utils)
        
        return summary
    
    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history = []
    
    def __del__(self):
        """Cleanup NVML on deletion"""
        if self.nvml_available:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass
