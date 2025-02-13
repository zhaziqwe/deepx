package deepx

import "fmt"

type Shape struct {
	shape  []int
	stride []int
	ndim   int
	size   int
}

func NewTensorShape(shape []int) (s Shape) {
	s.ndim = len(shape)
	s.shape = make([]int, len(shape))
	copy(s.shape, shape)
	s.stride = make([]int, len(shape))
	s.stride[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		s.stride[i] = s.stride[i+1] * shape[i+1]
	}
	s.size = s.stride[0] * shape[0]
	return s
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
	graph *Graph      // 所属计算图
	node  *TensorNode // 对应的计算图节点
}

func (t *Tensor) Add(other *Tensor) *Tensor {
	// 创建新Tensor（继承graph）
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape)
	// 添加操作节点
	op := t.graph.AddOp("add", OpAdd, t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}

func (t *Tensor) Sub(other *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape)
	// 添加操作节点
	op := t.graph.AddOp("sub", OpSub, t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}
func (t *Tensor) Mul(other *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape)
	// 添加操作节点
	op := t.graph.AddOp("mul", OpMul, t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}
func (t *Tensor) Div(other *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape)
	// 添加操作节点
	op := t.graph.AddOp("div", OpDiv, t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}
func (t *Tensor) Matmul(other *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape)
	op := t.graph.AddOp("matmul", OpMatmul, t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}
