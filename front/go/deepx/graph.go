package deepx

import (
	"fmt"
	"strings"
)

type Node struct {
	name    string
	ntype   NodeType
	inputs  []*Node
	outputs []*Node
	tensor  *Tensor // 对于 NodeTensor 类型，存储实际的张量数据
	op      OpType  // 对于 NodeOp 类型，存储操作类型
}

type NodeType int

const (
	NodeTensor NodeType = iota
	NodeOp
	NodeConstArg
)

type Graph struct {
	nodes         []*Node
	tensorCounter int // 新增计数器字段
}

// 创建新图
func NewGraph() *Graph {
	return &Graph{
		nodes: make([]*Node, 0),
	}
}

// 添加张量节点
func (g *Graph) AddTensor(name string, tensor *Tensor) *Node {
	if name == "" {
		name = fmt.Sprintf("tensor_%d", g.tensorCounter)
		g.tensorCounter++
	}
	node := &Node{
		name:    name,
		ntype:   NodeTensor,
		tensor:  tensor,
		inputs:  make([]*Node, 0),
		outputs: make([]*Node, 0),
	}
	g.nodes = append(g.nodes, node)
	return node
}

// 添加操作节点
func (g *Graph) AddOp(name string, opType OpType, inputs ...*Node) *Node {
	node := &Node{
		name:    name,
		ntype:   NodeOp,
		op:      opType,
		inputs:  inputs,
		outputs: make([]*Node, 0),
	}

	// 设置输入节点的输出连接
	for _, input := range inputs {
		input.outputs = append(input.outputs, node)
	}

	g.nodes = append(g.nodes, node)
	return node
}

// 前向计算
func (g *Graph) Forward(node *Node) *Tensor {
	if node.ntype == NodeTensor {
		return node.tensor
	}

	// 获取输入张量
	inputTensors := make([]*Tensor, len(node.inputs))
	for i, input := range node.inputs {
		inputTensors[i] = g.Forward(input)
	}

	// 根据操作类型执行计算
	switch node.op {
	case OpAdd:
		return inputTensors[0].Add(inputTensors[1])
	case OpSub:
		return inputTensors[0].Sub(inputTensors[1])
	case OpMul:
		return inputTensors[0].Mul(inputTensors[1])
	case OpDiv:
		return inputTensors[0].Div(inputTensors[1])
	case OpMatmul:
		return inputTensors[0].Matmul(inputTensors[1])
	default:
		return nil
	}
}
func (g *Graph) ToDOT() string {
	var builder strings.Builder
	builder.WriteString("digraph computational_graph {\n")
	builder.WriteString("  rankdir=LR;\n")
	builder.WriteString("  node [shape=record];\n")

	// 遍历所有节点
	for _, node := range g.nodes {
		// 根据节点类型设置不同样式
		switch node.ntype {
		case NodeTensor:
			builder.WriteString(fmt.Sprintf("  \"%p\" [label=\"%s%v\", shape=ellipse];\n",
				node, node.name, node.tensor.Shape))
		case NodeOp:
			builder.WriteString(fmt.Sprintf("  \"%p\" [label=\"%s\", shape=rectangle];\n",
				node, node.name))
		case NodeConstArg:
			builder.WriteString(fmt.Sprintf("  \"%p\" [label=\"const%s\", shape=diamond];\n",
				node, node.name))
		}
	}

	// 添加边连接
	for _, node := range g.nodes {
		for _, output := range node.outputs {
			builder.WriteString(fmt.Sprintf("  \"%p\" -> \"%p\"", node, output))

			// // 如果是操作节点，显示操作类型
			// if output.ntype == NodeOp {
			// 	builder.WriteString(fmt.Sprintf(" [label=\"%s\"]", opTypeToString(output.op)))
			// }
			builder.WriteString(";\n")
		}
	}

	builder.WriteString("}\n")
	return builder.String()
}
