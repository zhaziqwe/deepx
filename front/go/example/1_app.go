package main

import (
	"os"

	"github.com/array2d/deepx/front/go/deepx"
)

type Module1 struct {
	g *deepx.Graph
}

func (m *Module1) Forward() *deepx.Tensor {

	// 创建输入节点
	x_node := m.g.AddTensor("", deepx.DtypeFloat32, []int{1, 2, 3})
	w_node := m.g.AddTensor("", deepx.DtypeFloat32, []int{3, 4, 5})

	// 自动构建计算图
	y := x_node.Tensor().Matmul(w_node.Tensor())

	b_node := m.g.AddTensor("", deepx.DtypeFloat32, []int{1, 4, 5})
	z := y.Add(b_node.Tensor())

	return z
}

func main() {
	module := &Module1{
		g: deepx.NewGraph(),
	}
	module.Forward()

	dot := module.g.ToDOT()
	os.WriteFile("1_app.dot", []byte(dot), 0644)
}
