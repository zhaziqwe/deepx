from safetensors import safe_open
from deepx.tensor import Tensor, DeviceType
import numpy as np
import os
import json

class SafeTensorLoader:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.config = self._load_config()
        
    def _load_config(self):
        """加载模型配置"""
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
        
    def load(self):
        """加载safetensor模型文件"""
        tensors = {}
        metadata = {}
        
        model_path = os.path.join(self.model_dir, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
            
        with safe_open(model_path, framework="numpy") as f:
            metadata = f.metadata() if hasattr(f, 'metadata') else {}
            for key in f.keys():
                tensor = f.get_tensor(key)
                tensors[key] = self._convert_to_deepx_tensor(tensor, key)
                
        # 添加配置信息到metadata
        metadata["model_config"] = self.config
        return tensors, metadata
    
    def _convert_to_deepx_tensor(self, np_tensor, name):
        """将numpy数组转换为DeepX的Tensor"""
        return Tensor(data=np_tensor, name=name, shape=np_tensor.shape)

class SafeTensorSaver:
    def __init__(self, tensors, metadata=None):
        self.tensors = tensors
        self.metadata = metadata or {}
        
    def save(self, save_path):
        """保存模型到safetensor格式"""
        tensor_dict = {}
        for name, tensor in self.tensors.items():
            if isinstance(tensor, Tensor):
                tensor_dict[name] = tensor.data
            else:
                tensor_dict[name] = tensor
                
        from safetensors.numpy import save_file
        save_file(tensor_dict, save_path, metadata=self.metadata) 