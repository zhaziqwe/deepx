from deepx.tensor import Tensor
 
def main():
    # 创建计算图
    x = Tensor.placeholder(name="input", shape=(32, 784))
    w = Tensor.placeholder(name="weight", shape=(784, 10))
    b = Tensor.placeholder(name="bias", shape=(10,))
    
    # 构建简单的神经网络
    hidden = x @ w + b
    output = hidden.relu()
    
    # 导出计算图为DOT格式
    dot = output.node.graph.to_dot()
    dot.render("simple_network", format="png", cleanup=True)
    
    print("计算图已保存为 simple_network.png")

if __name__ == "__main__":
    main() 