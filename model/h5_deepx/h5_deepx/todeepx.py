import h5py
import numpy as np
import yaml
import os   
import sys

output="output/"

def traverse_h5_group(group, keypath, callback):
    """
    遍历 H5 组，调用回调函数处理每个数据集。
    
    :param group: h5py.Group 对象
    :param keypath: 当前的键路径
    :param callback: 处理数据集的回调函数
    """
    for key, item in group.items():
        if key[-1]!='/':
            current_keypath = f"{keypath}.{key}"  # 构建当前的键路径
        else:
            current_keypath = f"{keypath}{key}"  # 构建当前的键路径
        if isinstance(item, h5py.Group):
            # 如果是组，递归遍历
            traverse_h5_group(item, current_keypath, callback)
        else:
            # 如果是数据集，调用回调函数
            callback(current_keypath, item)
def save_dataset(keypath, dataset):
    """
    处理数据集，提取类型、形状和数据，并保存。
    
    :param keypath: 数据集的键路径
    :param dataset: h5py.Dataset 对象
    """
    weight_values = dataset[()]  # 获取数据
    weight_shape = weight_values.shape  # 获取形状
    weight_type = weight_values.dtype  # 获取数据类型

    # 保存形状为 YAML 格式
    shape_info = {
        'type': str(weight_type),  # 数据类型
        'shape':list(weight_shape) # 转换为列表以便于 YAML 序列化
    }
    shapepath = f"{output}.{keypath}.shape"
    shapepath_dir = os.path.dirname(shapepath)  # 获取目录路径
    if shapepath_dir and not os.path.exists(shapepath_dir):
        os.makedirs(shapepath_dir)  # 创建目录
    with open(shapepath, 'w') as shape_file:
        yaml.dump(shape_info, shape_file, default_flow_style=False)

    # 保存数据为原始字节数组
    datapath = f"{output}.{keypath}.data"
    datapath_dir = os.path.dirname(datapath)  # 获取目录路径
    if datapath_dir and not os.path.exists(datapath_dir):
        os.makedirs(datapath_dir)  # 创建目录
    with open(datapath, 'wb') as data_file:
        data_file.write(weight_values.tobytes())
def h5_graph(h5_file):
    # 打开 H5 文件
    with h5py.File(h5_file, 'r') as f:
        print("H5 file contents:")
        f.visit(print)  # 打印所有对象的名称

        if 'model_config' not in f:
            print("No model config")
        else:
            model_config = f['model_config'][()]
            # 提取模型结构
            model_layers = []
            for layer in model_config['layers']:
                layer_info = {
                    'name': layer['name'],
                    'type': layer['class_name'],
                    'config': layer['config']
                }
            model_layers.append(layer_info)
            output_file=output+"/graph.yaml"
            with open(output_file, 'w') as outfile:
                yaml.dump(model_layers, outfile, default_flow_style=False)

def h5_params(h5_file):
    with h5py.File(h5_file, 'r') as f:
        # 提取权重
        if 'model_weights' not in f:
            print("No model weights")
        else:
            weights = f['model_weights']
            traverse_h5_group(weights, 'tensormap/', save_dataset)

if __name__ == "__main__":
    h5_file=sys.argv[1]
    output=h5_file.replace(".h5","")
    if len(sys.argv)>2:
        output=sys.argv[2]
    h5_graph(h5_file)
    h5_params(h5_file)
    print(f"Model layers and weights saved to {output}")
