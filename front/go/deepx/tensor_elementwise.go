package deepx

func init() {
	RegistOpType("add", "+")
	RegistOpType("sub", "-")
	RegistOpType("mul", "✖")
	RegistOpType("div", "÷")
	RegistOpType("scale", "×1/√d")
}

func (t *Tensor) Add(other *Tensor) *Tensor {
	// 创建新Tensor（继承graph）
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)
	// 添加操作节点
	op := t.graph.AddOp("add", t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}

func (t *Tensor) Sub(other *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)
	// 添加操作节点
	op := t.graph.AddOp("sub", t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}
func (t *Tensor) Mul(other *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)
	// 添加操作节点
	op := t.graph.AddOp("mul", t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}
func (t *Tensor) Div(other *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)
	// 添加操作节点
	op := t.graph.AddOp("div", t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}

// Scale 对张量进行缩放操作
// 在注意力机制中，通常用于缩放点积注意力分数，防止其过大导致softmax梯度消失
// 缩放因子通常为 1/sqrt(d_k)，其中d_k是注意力头的维度
//
// 数学表达:
// - 设输入张量为X，缩放因子为s
// - 输出张量Y = s * X
//
// 在注意力中的应用:
// 1. Q和K的点积得到注意力分数: score = Q * K^T
// 2. 缩放分数: scaled_score = score / sqrt(d_k)
//   - d_k 是注意力头的维度
//   - 这样可以让方差保持在1左右
//   - 防止softmax输入过大，导致梯度消失
func (t *Tensor) Scale(factor float32) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)
	// 使用 ×1/√d 作为操作符号，直观显示缩放操作
	op := t.graph.AddOp("scale", t.node)

	// 添加缩放因子作为常量参数
	factor_node := t.graph.AddConstArg(t.node.Name() + ".scale_factor")
	factor_node.SetFloat(float64(factor))
	op.AddInput("factor", factor_node)

	result.AddInput(op.name, op)
	return result.tensor
}
