package deepx

import "fmt"

type Shape struct {
	shape  []int
	stride []int
	ndim   int
	size   int
}

func (s Shape) String() string {
	return fmt.Sprintf("%v", s.shape)
}

type Dtype int

const (
	DtypeInt8 Dtype = iota
	DtypeInt16
	DtypeInt32
	DtypeInt64
	DtypeUint8
	DtypeFloat16
	DtypeFloat32
	DtypeFloat64
)

type Tensor struct {
	Data  []byte
	Dtype Dtype
	Shape Shape
	graph *Graph // 所属计算图
	node  *Node  // 对应的计算图节点
}

func NewTensor(g *Graph, dtype Dtype) *Tensor {
	t := &Tensor{
		Dtype: dtype,
		graph: g,
	}
	// 自动创建对应的Tensor节点
	t.node = g.AddTensor("", t)
	return t
}

func (t *Tensor) Add(other *Tensor) *Tensor {
	// 创建新Tensor（继承graph）
	result := NewTensor(t.graph, t.Dtype)
	// 添加操作节点
	t.graph.AddOp("add", OpAdd, t.node, other.node).outputs = []*Node{result.node}
	return result
}

func (t *Tensor) Sub(other *Tensor) *Tensor {
	result := NewTensor(t.graph, t.Dtype)
	// 添加操作节点
	t.graph.AddOp("sub", OpSub, t.node, other.node).outputs = []*Node{result.node}
	return result
}
func (t *Tensor) Mul(other *Tensor) *Tensor {
	result := NewTensor(t.graph, t.Dtype)
	// 添加操作节点
	t.graph.AddOp("mul", OpMul, t.node, other.node).outputs = []*Node{result.node}
	return result
}
func (t *Tensor) Div(other *Tensor) *Tensor {
	result := NewTensor(t.graph, t.Dtype)
	// 添加操作节点
	t.graph.AddOp("div", OpDiv, t.node, other.node).outputs = []*Node{result.node}
	return result
}
func (t *Tensor) Matmul(other *Tensor) *Tensor {
	result := NewTensor(t.graph, t.Dtype)
	t.graph.AddOp("matmul", OpMatmul, t.node, other.node).outputs = []*Node{result.node}
	return result
}
