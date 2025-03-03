from deepx.tensor import Tensor
from deepx.autograd.graph import Graph
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send

def newtensor(t:Tensor):
    graph = Graph.get_default()
    t._graph = graph
    t._node=graph.add_tensor("",t=t)
    if t.graph.eager:
        ir2=DeepxIR("newtensor", t.dtype, t.shape, [t._node.name])
        send(ir2)
def copytensor(t:Tensor,out:Tensor):
    graph = Graph.get_default()
    out.node.add_input(t.node)
    if t.graph.eager:
        ir2=DeepxIR("copytensor", t.dtype, [t.node.name], [out.node.name])
        send(ir2)