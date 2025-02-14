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
func (g *Graph) ToDOT() string {
	var builder strings.Builder
	builder.WriteString("digraph computational_graph {\n")
	builder.WriteString("  rankdir=TB;\n")
	builder.WriteString("  node [shape=record];\n")

	// 遍历所有节点
	for _, node := range g.nodes {
		builder.WriteString(fmt.Sprintf(`"%p" `, node))
		switch node.Ntype() {
		case NodeTensor:
			// 张量节点：显示形状和梯度信息
			builder.WriteString(fmt.Sprintf(`[label= "%s \n %v`, node.Name(), node.(*TensorNode).Tensor().Shape))
			// if node.(*TensorNode).Tensor().requiresGrad {
			// 	builder.WriteString(`\n require_grad`)
			// }
			builder.WriteString(`"`)
			builder.WriteString(",shape=box")
			builder.WriteString(",labeljust=l")
			builder.WriteString(",color=skyblue")
			builder.WriteString(",style=filled")
			builder.WriteString(",fillcolor=aliceblue")
			builder.WriteString(",fontname=\"Sans-Serif\"")

		case NodeOp:
			// 操作节点：突出显示操作类型
			opNode := node.(*OpNode)
			builder.WriteString(fmt.Sprintf(`[label="%s"`, opNode.Shortchar()))
			builder.WriteString(",shape=box")
			builder.WriteString(",style=filled")
			builder.WriteString(",fillcolor=lightgray")
			builder.WriteString(",color=darkslategray")
			builder.WriteString(",fontname=\"Courier Bold\"")

		case NodeConstArg:
			// 常量参数节点：显示参数值
			constNode := node.(*ConstArgNode)
			var valueStr string
			switch constNode.argType {
			case ArgTypeInt:
				valueStr = fmt.Sprintf("%d", constNode.Int())
			case ArgTypeFloat:
				valueStr = fmt.Sprintf("%.2f", constNode.Float()) + "f"
			case ArgTypeString:
				valueStr = constNode.String()
			}
			builder.WriteString(fmt.Sprintf(`[label="%s"`, valueStr))
			builder.WriteString(",shape=diamond")
			builder.WriteString(",style=filled")
			builder.WriteString(",fillcolor=lightyellow")
			builder.WriteString(",color=goldenrod")
			builder.WriteString(",fontname=\"Sans-Serif\"")
		}
		builder.WriteString("];\n")
	}

	// 添加边连接，为边也添加样式
	for _, node := range g.nodes {
		for _, input := range node.Inputs() {
			builder.WriteString(fmt.Sprintf(`  "%p" -> "%p"`, input, node))
			builder.WriteString(`[color=gray40,penwidth=1.2,arrowsize=0.8]`)
			builder.WriteString(";\n")
		}
	}

	builder.WriteString("}\n")
	return builder.String()
}
