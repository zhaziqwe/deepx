package deepx

func (t *Tensor) Mean(dims []int) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)

	axis_node := t.graph.AddConstArg("axis")
	axis_node.SetInt(axis)

	op := t.graph.AddOp("mean", t.node, axis_node)
	result.AddInput(op.name, op)

	return result.tensor
}

func (t *Tensor) Sum(axis int) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)

	axis_node := t.graph.AddConstArg("axis")
	axis_node.SetInt(axis)

	op := t.graph.AddOp("sum", t.node, axis_node)
	result.AddInput(op.name, op)

	return result.tensor
}

func (t *Tensor) Prod(axis int) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)

	axis_node := t.graph.AddConstArg("axis")
	axis_node.SetInt(axis)

	op := t.graph.AddOp("prod", t.node, axis_node)
	result.AddInput(op.name, op)

	return result.tensor
}

func (t *Tensor) Max(axis int) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)

	axis_node := t.graph.AddConstArg("axis")
	axis_node.SetInt(axis)

	op := t.graph.AddOp("max", t.node, axis_node)
	result.AddInput(op.name, op)

	return result.tensor
}

func (t *Tensor) Min(axis int) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)

	axis_node := t.graph.AddConstArg("axis")
	axis_node.SetInt(axis)

	op := t.graph.AddOp("min", t.node, axis_node)
	result.AddInput(op.name, op)

	return result.tensor
}
