package deepx

type OpNode struct {
	name   string
	ntype  NodeType
	inputs map[string]Node
	op     OpType
}

func NewOpNode(name string, op OpType) *OpNode {
	return &OpNode{
		name:   name,
		ntype:  NodeOp,
		inputs: make(map[string]Node),
		op:     op,
	}
}
func (n *OpNode) Ntype() NodeType {
	return n.ntype
}
func (n *OpNode) Name() string {
	return n.name
}
func (n *OpNode) Input(name string) Node {
	return n.inputs[name]
}
func (n *OpNode) Inputs() map[string]Node {
	return n.inputs
}
func (n *OpNode) AddInput(name string, input Node) {
	n.inputs[name] = input
}
func (n *OpNode) RemoveInput(name string) {
	delete(n.inputs, name)
}
func (n *OpNode) Op() OpType {
	return n.op
}
