from typing import Union
from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send
from deepx.autograd import OpNode

OpNode.register("transpose")
def transpose(t: Tensor,dimorder:list[int]=None,inplace:bool=False,out:Union[Tensor,str]='')->Tensor:
    if dimorder is None:
        dimorder=list(range(t.ndimension))
    outtensor=None
    if inplace:
        outtensor=t
    else:
        if isinstance(out,str):
            outtensor=Tensor(shape=t.Shape.transpose(dimorder), dtype=t.dtype, device=t.device)
            outtensor.addtograph(out)
        else:
            outtensor=out
    vectornode=t.graph.add_vector("",dimorder)
    opnode = t.graph.add_op("transpose")
    opnode.add_input(t._node)
    opnode.add_input(vectornode)

    outtensor.node.add_input(opnode)
    if t.graph.eager:
        ir=DeepxIR("transpose",'',[t._node.name,*map(str, dimorder)], [outtensor._node.name])
        send(ir)

    return outtensor
 
OpNode.register("reshape")
def reshape(t:Tensor,shape:list[int],inplace:bool=False,out:Union[Tensor,str]='')->Tensor:
    outtensor=None
    if inplace:
        outtensor=t
        from deepx  import Shape
        outtensor._shape=Shape(shape)
    else:
        if isinstance(out,str):
            outtensor=Tensor(shape=shape, dtype=t.dtype, device=t.device)
            outtensor.addtograph(out)
        else:
            outtensor=out
    opnode=t.graph.add_op("reshape")
    opnode.add_input(t.node)
    opnode.add_input(t.graph.add_vector("",shape))
    outtensor.node.add_input(opnode)
    if t.graph.eager:
        ir=DeepxIR("reshape",'',[t.node.name,*map(str, shape)], [outtensor.node.name])
        send(ir)
    return outtensor
    