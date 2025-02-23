from safetensors import safe_open
import numpy as np
import os
import json
import yaml
import argparse
import shutil


class TensorInfo:
    def __init__(self, dtype, ndim, shape, size, strides=None):
        self.dtype = dtype  # 数据精度类型，如"float32"
        self.ndim = ndim  # 维度数
        self.shape = shape  # 形状元组
        self.size = size  # 总元素数量
        self.strides = strides  # 步长数组（可选）


class Tensor:
    def __init__(self, data, tensorinfo):
        """
        :param data: bytes 原始字节数据
        :param tensorinfo: TensorInfo 元数据
        """
        if not isinstance(tensorinfo, TensorInfo):
            raise TypeError("tensorinfo必须是TensorInfo实例")

        self.data = data
        self.tensorinfo = tensorinfo
        self.graph = None  # 所属计算图（与Go版对齐）
        self.node = None  # 对应计算图节点
        self.requires_grad = False  # 是否需要梯度

    def __repr__(self):
        return f"Tensor(dtype={self.tensorinfo.dtype}, shape={self.tensorinfo.shape})"


class SafeTensorExporter:
    def __init__(self, model_dir, output_dir):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.config = self._load_config()
        self.dtype_map = {  # 添加数据类型映射表
            'BF16': 'float32',  # 将bfloat16转换为float32保存
            'F16': 'float16',
            'F32': 'float32'
        }

    def _load_config(self):
        """加载模型配置"""
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def export(self):
        """导出safetensor模型到指定目录"""
        model_path = os.path.join(self.model_dir, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        # 修改为使用PyTorch框架加载
        with safe_open(model_path, framework="pt") as f:  # 改为pt框架
            for key in f.keys():
                tensor = f.get_tensor(key)
                self._save_tensor(key, tensor)

        # 保存全局配置
        self._save_config()
        self._copy_tokenizer_files()

    def _save_tensor(self, name, tensor):
        """保存单个张量的元数据和二进制数据"""
        # 将名称中的点转换为下划线，并创建统一路径
        base_path = os.path.join(self.output_dir, "tensors", name)
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        # 处理bfloat16类型
        dtype_str = str(tensor.dtype).replace("torch.", "")
        if dtype_str == "bfloat16":
            tensor = tensor.float()
            dtype_str = "float32"
        
        # 更新后的类型处理
        shape_info = {
            'dtype': self.dtype_map.get(dtype_str.upper(), dtype_str),
            'shape': list(tensor.shape)
        }

        # 保存为numpy格式
        np_tensor = tensor.numpy().astype(shape_info['dtype'])
        with open(f"{base_path}.data", 'wb') as f:
            f.write(np_tensor.tobytes())
        
        with open(f"{base_path}.shape", 'w') as f:
            yaml.dump(shape_info, f, default_flow_style=False)

    def _save_config(self):
        """保存模型全局配置"""
        config_path = os.path.join(self.output_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump({
                'model_config': self.config,
                'format_version': 'deepx'
            }, f, default_flow_style=False)

    def _copy_tokenizer_files(self):
        """复制tokenizer相关文件到输出目录"""
        required_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json"
        ]
        
        for filename in required_files:
            src = os.path.join(self.model_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(self.output_dir, filename))


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

        with safe_open(model_path, framework="pt") as f:  # 修改为pt框架
            metadata = f.metadata() if hasattr(f, 'metadata') else {}
            for key in f.keys():
                pt_tensor = f.get_tensor(key).cpu().detach()  # 获取PyTorch张量

                # 构造TensorInfo
                tensor_info = TensorInfo(
                    dtype=str(pt_tensor.dtype).replace("torch.", ""),
                    ndim=pt_tensor.ndim,
                    shape=tuple(pt_tensor.shape),
                    size=pt_tensor.numel(),
                    strides=pt_tensor.stride() if pt_tensor.is_contiguous() else None
                )

                # 转换为字节流（保持内存对齐）
                byte_buffer = pt_tensor.numpy().tobytes() if pt_tensor.device == "cpu" \
                    else pt_tensor.cpu().numpy().tobytes()

                tensors[key] = Tensor(byte_buffer, tensor_info)

        metadata["model_config"] = self.config
        return tensors, metadata


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

    # 使用示例


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Safetensor模型转换工具')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='输入目录路径，包含model.safetensors和config.json')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录路径，转换后的DeepX格式数据将保存于此')

    args = parser.parse_args()

    exporter = SafeTensorExporter(
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    try:
        exporter.export()
        print(f"转换成功！输出目录：{args.output_dir}")
    except Exception as e:
        print(f"转换失败：{str(e)}")
        exit(1)