package deepx

import (
	"fmt"
)

type Node interface {
	Ntype() NodeType
	Name() string
	Inputs() map[string]Node
	Input(name string) Node
	AddInput(name string, input Node)
	RemoveInput(name string)
}

type NodeType int

const (
	NodeTensor NodeType = iota
	NodeOp
	NodeConstArg
)

type Graph struct {
	nodes           []Node
	tensorCounter   int
	constArgCounter int
	enableGrad      bool
}

// 创建新图
func NewGraph() *Graph {
	return &Graph{
		nodes: make([]Node, 0),
	}
}

// 添加张量节点
func (g *Graph) AddTensor(name string, dtype Dtype, shape []int, requiresGrad bool, inputs ...Node) *TensorNode {
	if name == "" {
		name = fmt.Sprintf("tensor_%d", g.tensorCounter)
		g.tensorCounter++
	}
	node := NewTensorNode(name)
	node.SetTensor(&Tensor{
		Dtype:        dtype,
		graph:        g,
		Shape:        NewTensorShape(shape),
		node:         node,
		requiresGrad: requiresGrad,
	})
	for _, input := range inputs {
		node.AddInput(input.Name(), input)
	}
	g.nodes = append(g.nodes, node)
	return node
}

// 添加操作节点
func (g *Graph) AddOp(name string, inputs ...Node) *OpNode {
	node := NewOpNode(name)
	for _, input := range inputs {
		node.AddInput(input.Name(), input)
	}
	g.nodes = append(g.nodes, node)
	return node
}
func (g *Graph) AddConstArg(name string) *ConstArgNode {
	if name == "" {
		name = fmt.Sprintf("const_%d", g.constArgCounter)
		g.constArgCounter++
	}
	node := NewConstArgNode(name)
	g.nodes = append(g.nodes, node)
	return node
}

// 前向计算
func (g *Graph) Forward(node Node) *Tensor {
	if node.Ntype() == NodeTensor {
		return node.(*TensorNode).Tensor()
	}
	return nil
}
