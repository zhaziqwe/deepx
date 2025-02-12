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
	x := deepx.NewTensor(m.g, deepx.DtypeFloat32)
	w := deepx.NewTensor(m.g, deepx.DtypeFloat32)

	// 自动构建计算图
	y := x.Matmul(w)
	z := y.Add(deepx.NewTensor(m.g, deepx.DtypeFloat32))

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
