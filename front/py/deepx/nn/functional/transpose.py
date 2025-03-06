from typing import Union
from deepx.tensor import Tensor
from deepx.nn.deepxir import DeepxIR
from deepx.scheduler import send
from deepx.autograd import OpNode


#transpose support exchange 2 dimension,or reorder all dimension
OpNode.register("transpose")
def transpose(t: Tensor,dimorder:list[int]=None,inplace:bool=False,out:Union[Tensor,str]='')->Tensor:
    if dimorder is None:
        return t
    
    # 处理负数索引并去重
    ndim = t.ndimension
    dimorder = [d % ndim for d in dimorder]
    unique_dims = list(dict.fromkeys(dimorder)) 


    final_dimorder=[]
    if len(unique_dims) == 2:
        # 模式1：交换两个维度
        d1, d2 = unique_dims
        final_dimorder = list(range(ndim))
        final_dimorder[d1], final_dimorder[d2] = final_dimorder[d2], final_dimorder[d1]
    elif len(unique_dims) == ndim:
        # 模式2：全排列
        final_dimorder = unique_dims
    else:
        raise ValueError(f"维度参数不合法，支持两种模式：1.交换两个维度 2.全排列维度顺序，当前输入维度数：{len(unique_dims)}，张量维度数：{ndim}")


    outtensor=None
    if inplace:
        outtensor=t
    else:
        if isinstance(out,str):
            outtensor=Tensor(shape=t.Shape.transpose(final_dimorder), dtype=t.dtype, device=t.device)
            outtensor.addtograph(out)
        else:
            outtensor=out
    vectornode=t.graph.add_vector("",final_dimorder)
    opnode = t.graph.add_op("transpose")
    opnode.add_input(t._node)
    opnode.add_input(vectornode)

    outtensor.node.add_input(opnode)
    if t.graph.eager:
        ir=DeepxIR("transpose",'',[t._node.name,*map(str, final_dimorder)], [outtensor._node.name])
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
    