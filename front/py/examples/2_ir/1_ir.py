from deepx.tensor.deepxir import DeepxIR
# 正向传播示例
op = DeepxIR(
    name="add",
    args=["t1", "t2"], 
    returns=["t3"],
    grad=True,
    args_grad=["t1_grad", "t2_grad"],
    returns_grad=["t3_grad"]
)
print(op.to_ir("float32"))  
# 输出: add@float32 t1 t2 -> t3

# 反向传播示例
print(op.to_grad_ir("float32"))  
# 输出: add@float32 t1(t1_grad) t2(t2_grad) <- t3(t3_grad)

# 标量操作示例
scalar_op = DeepxIR(
    name="scalar",
    args=["t1", "a1"],  # a1为参数
    returns=["t3"],
    grad=True,
    args_grad=["t1_grad",""],
    returns_grad=["t3_grad"]
)
print(scalar_op.to_ir("float32"))
# 输出: add_scalar@float32 t1 a1 -> t3
print(scalar_op.to_grad_ir("float32"))
# 输出: add_scalar@float32 t1(t1_grad) a1 <- t3(t3_grad)