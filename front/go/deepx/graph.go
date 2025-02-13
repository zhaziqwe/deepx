package deepx

import (
	"fmt"
	"strings"
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
	tensorCounter   int // 新增计数器字段
	opCounter       int // 新增计数器字段
	constArgCounter int // 新增计数器字段

}

// 创建新图
func NewGraph() *Graph {
	return &Graph{
		nodes: make([]Node, 0),
	}
}

// 添加张量节点
func (g *Graph) AddTensor(name string, dtype Dtype, shape []int, inputs ...Node) *TensorNode {
	if name == "" {
		name = fmt.Sprintf("tensor_%d", g.tensorCounter)
		g.tensorCounter++
	}
	node := NewTensorNode(name)
	node.SetTensor(&Tensor{
		Dtype: dtype,
		graph: g,
		Shape: NewTensorShape(shape),
		node:  node,
	})
	for _, input := range inputs {
		node.AddInput(input.Name(), input)
	}
	g.nodes = append(g.nodes, node)
	return node
}

// 添加操作节点
func (g *Graph) AddOp(name string, opType OpType, inputs ...Node) *OpNode {
	if name == "" {
		name = fmt.Sprintf("op_%d", g.opCounter)
		g.opCounter++
	}
	node := NewOpNode(name, opType)
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
func (g *Graph) ToDOT() string {
	var builder strings.Builder
	builder.WriteString("digraph computational_graph {\n")
	builder.WriteString("  rankdir=LR;\n")
	builder.WriteString("  node [shape=record];\n")

	// 遍历所有节点
	for _, node := range g.nodes {
		// 根据节点类型设置不同样式
		switch node.Ntype() {
		case NodeTensor:
			builder.WriteString(fmt.Sprintf("  \"%p\" [label=\"%s%v\", shape=ellipse];\n",
				node, node.Name(), node.(*TensorNode).Tensor().Shape))
		case NodeOp:
			builder.WriteString(fmt.Sprintf("  \"%p\" [label=\"%s\", shape=rectangle];\n",
				node, node.Name()))
		case NodeConstArg:
			builder.WriteString(fmt.Sprintf("  \"%p\" [label=\"const%s\", shape=diamond];\n",
				node, node.Name()))
		}
	}

	// 添加边连接
	for _, node := range g.nodes {
		for _, input := range node.Inputs() {
			builder.WriteString(fmt.Sprintf("  \"%p\" -> \"%p\"", input, node))
			builder.WriteString(";\n")
		}
	}

	builder.WriteString("}\n")
	return builder.String()
}
