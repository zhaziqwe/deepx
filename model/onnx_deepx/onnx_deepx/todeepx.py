import onnx

def extract_onnx_info(onnx_file):
    # 加载 ONNX 模型
    model = onnx.load(onnx_file)

    # 提取模型信息
    model_info = {
        "ir_version": model.ir_version,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "domain": model.domain,
        "model_version": model.model_version,
        "doc_string": model.doc_string,
        "graph": {
            "name": model.graph.name,
            "input": [input.name for input in model.graph.input],
            "output": [output.name for output in model.graph.output],
            "nodes": []
        }
    }

    # 提取节点信息
    for node in model.graph.node:
        model_info["graph"]["nodes"].append({
            "name": node.name,
            "op_type": node.op_type,
            "inputs": node.input,
            "outputs": node.output,
            "attributes": {attr.name: attr for attr in node.attribute}
        })

    return model_info