from deepx.tensor import Tensor
from deepx.autograd import OpNode
from deepx.nn import DeepxIR
from deepx.scheduler import send

OpNode.register("print")
def printtensor(t:Tensor):
    ir=DeepxIR("print",'', [t.node.name], [])
    send(ir)
    return ''

