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
	Data         []byte
	Dtype        Dtype
	Shape        Shape
	graph        *Graph      // 所属计算图
	node         *TensorNode // 对应的计算图节点
	requiresGrad bool
}

func (t *Tensor) Matmul(other *Tensor) *Tensor {
	result := t.graph.AddTensor("", t.Dtype, t.Shape.shape, t.requiresGrad)
	op := t.graph.AddOp("matmul", "matmul", t.node, other.node)
	result.AddInput(op.name, op)
	return result.tensor
}
