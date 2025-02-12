package deepx

type Module interface {
	Forward() *Tensor
}

func NewModule(graph *Graph) Module {
	return &Module{graph: graph}
}
