package deepx

func init() {
	RegistOpType("layernorm", "LN")
	RegistOpType("batchnorm", "BN")
	RegistOpType("instancenorm", "IN")
	RegistOpType("groupnorm", "GN")
}

func (t *Tensor) LayerNorm(weight, bias *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)

	op := t.graph.AddOp("layernorm", t.node, weight.node, bias.node)
	result.AddInput(op.name, op)

	return result.tensor
}

func (t *Tensor) BatchNorm(weight, bias, running_mean, running_var *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)

	op := t.graph.AddOp("batchnorm", t.node, weight.node, bias.node,
		running_mean.node, running_var.node)
	result.AddInput(op.name, op)

	return result.tensor
}

func (t *Tensor) InstanceNorm(weight *Tensor, bias *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)

	op := t.graph.AddOp("instancenorm", t.node, weight.node, bias.node)
	result.AddInput(op.name, op)

	return result.tensor
}

func (t *Tensor) GroupNorm(weight, bias *Tensor, num_groups int) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)

	num_groups_node := t.graph.AddConstArg("num_groups")
	num_groups_node.SetInt(num_groups)

	op := t.graph.AddOp("groupnorm", t.node, weight.node, bias.node, num_groups_node)
	result.AddInput(op.name, op)

	return result.tensor
}
