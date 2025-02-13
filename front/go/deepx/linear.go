package deepx

type Linear struct {
	ModuleBase
}

func NewLinear(name string, in_features, out_features int) (m *Linear) {
	m = &Linear{
		ModuleBase: ModuleBase{
			g: NewGraph(),
		},
	}
	if name == "" {
		name = "linear"
	}
	m.name = name
	return m
}
func (m *Linear) Linear(input *Tensor) *Tensor {
	// 创建输入节点
	w_node := m.AddTensor("W", DtypeFloat32, []int{3, 4, 5})

	// 自动构建计算图
	y := input.Matmul(w_node.Tensor())

	b_node := m.AddTensor("b", DtypeFloat32, []int{1, 4, 5})
	z := y.Add(b_node.Tensor())
	return z
}
