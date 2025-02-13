package deepx

func (t *Tensor) Add(other *Tensor) *Tensor {
	// 创建新Tensor（继承graph）
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)
	// 添加操作节点
	op := t.graph.AddOp("add", "+", t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}

func (t *Tensor) Sub(other *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)
	// 添加操作节点
	op := t.graph.AddOp("sub", "-", t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}
func (t *Tensor) Mul(other *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)
	// 添加操作节点
	op := t.graph.AddOp("mul", "✖", t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}
func (t *Tensor) Div(other *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)
	// 添加操作节点
	op := t.graph.AddOp("div", "÷", t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}
