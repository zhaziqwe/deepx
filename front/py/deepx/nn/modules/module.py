from deepx.autograd import Graph

class Module:
    def __init__(self):
        self._graph =Graph.get_default_graph()

    def forward(self, *args, **kwargs):
        pass
