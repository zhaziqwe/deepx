package deepx

func init() {
	RegistOpType("softmax", "softmax")
}

func (t *Tensor) Softmax() *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)
	op := t.graph.AddOp("softmax", t.node)
	result.AddInput(op.name, op)
	return result.tensor
}
