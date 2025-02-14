from deepx.graph import Graph, Node
from .loader import SafeTensorLoader

class SafeTensorGraphBuilder:
    def __init__(self, model_dir):
        self.loader = SafeTensorLoader(model_dir)
        
    def build_graph(self):
        """从safetensor模型构建计算图"""
        tensors, metadata = self.loader.load()
        graph = Graph()
        
        # 创建所有tensor节点
        tensor_nodes = {}
        for name, tensor in tensors.items():
            node = Node("Tensor", name=name)
            node.set_attr("shape", str(tensor.shape))
            node.set_attr("dtype", str(tensor.data.dtype))
            tensor_nodes[name] = node
            graph.add_node(node)
            
        # 从配置中提取模型结构信息
        model_config = metadata.get("model_config", {})
        self._build_transformer_structure(graph, tensor_nodes, model_config)
            
        return graph, tensors, metadata
    
    def _build_transformer_structure(self, graph, tensor_nodes, config):
        """根据Transformer结构构建连接关系"""
        num_layers = config.get("num_hidden_layers", 0)
        
        # 构建每一层的结构
        for layer_idx in range(num_layers):
            prefix = f"transformer.h.{layer_idx}."
            self._build_attention_layer(graph, tensor_nodes, prefix)
            self._build_mlp_layer(graph, tensor_nodes, prefix)
            
    def _build_attention_layer(self, graph, tensor_nodes, prefix):
        """构建注意力层的连接"""
        attn_weights = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "attn.c_attn.bias",
            "attn.c_proj.bias"
        ]
        
        for weight_name in attn_weights:
            full_name = prefix + weight_name
            if full_name in tensor_nodes:
                node = tensor_nodes[full_name]
                node.set_attr("layer_type", "attention")
                
    def _build_mlp_layer(self, graph, tensor_nodes, prefix):
        """构建MLP层的连接"""
        mlp_weights = [
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
            "mlp.c_fc.bias",
            "mlp.c_proj.bias"
        ]
        
        for weight_name in mlp_weights:
            full_name = prefix + weight_name
            if full_name in tensor_nodes:
                node = tensor_nodes[full_name]
                node.set_attr("layer_type", "mlp")

    def _build_connections(self, graph, tensor_nodes, structure):
        """根据结构信息构建节点间的连接"""
        try:
            structure_dict = eval(structure)
            for node_name, connections in structure_dict.items():
                if node_name in tensor_nodes:
                    node = tensor_nodes[node_name]
                    for input_name in connections.get("inputs", []):
                        if input_name in tensor_nodes:
                            node.add_input(tensor_nodes[input_name])
        except:
            pass  # 如果结构信息无效，就只保留独立的tensor节点 