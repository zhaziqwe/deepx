from typing import Optional, Union

from deepx.tensor import Tensor
from deepx.autograd.graph import OpNode
from deepx.nn.deepxir import DeepxIR    
from deepx.scheduler import send
from .elementwise import _A_b_elementwiseop_C

def reduceshape(inshape: Union[list[int], tuple[int]], 
               dims: Union[list[int], tuple[int]], 
               keepdim: bool) -> tuple[int]:
    """计算维度缩减后的形状
    
    Args:
        inshape: 输入形状，如(2,3,4)
        dims: 要缩减的维度列表，支持负数索引，如[-1]
        keepdim: 是否保留缩减后的维度为1
        
    Returns:
        缩减后的形状元组
        
    Example:
        >>> reduceshape((2,3,4), [1], True)
        (2, 1, 4)
        >>> reduceshape((2,3,4), [1], False)
        (2, 4)
    """
    ndim = len(inshape)
    # 处理负数维度
    normalized_dims = [d % ndim for d in dims]
    # 去重并排序
    unique_dims = sorted(set(normalized_dims))
    
    if keepdim:
        return tuple(1 if i in unique_dims else s 
                   for i, s in enumerate(inshape))
    else:
        return tuple(s for i, s in enumerate(inshape)
                   if i not in unique_dims)

def _A_v_reduceop_C(
        a:Tensor,
        dims: Union[list[int],tuple[int]] = [],
        keepdim:bool=False,
        op:str=None,
        out:Union[Tensor,str]='')->Tensor:
    
    if dims is None:
        dims=list(range(a.ndim))

    result=None
    if isinstance(out,str):
        resultshape=reduceshape(a.shape,dims,keepdim)
        result=Tensor(shape=resultshape, dtype=a.dtype, device=a.device)
        result.addtograph(out)
    else:
        result=out

    vector_node=a.graph.add_vector("",dims)
    opnode = a.graph.add_op(op)
    opnode.add_input(a.node)
    opnode.add_input(vector_node)
    result.node.add_input(opnode)

    if a.graph.eager:
        args = [*dims, "keepdim"] if keepdim else [*dims]
        varir=DeepxIR("argset",'int32', args, [vector_node.name])

        send(varir)
        ir=DeepxIR(op, a.dtype, [a.node.name,vector_node.name], [result.node.name])
        send(ir)
    return result

#max

OpNode.register("reduce_max")
def reduce_max(
        a:Tensor,
        dims:list[int] = None,
        keepdim=False,
        out:Union[Tensor,str]=''):
    return _A_v_reduceop_C(a,dims,keepdim,"max",out)
 
#min    
OpNode.register("reduce_min")
def reduce_min(
        a:Tensor,
        dims:list[int] = None,
        keepdim=False,
        out:Union[Tensor,str]=''):
    return _A_v_reduceop_C(a,dims,keepdim,"min",out)
    
 
#sum    
OpNode.register("sum")
def sum(
        a:Tensor,
        dims:Optional[Union[
            list[int],
            tuple[int],
            ]]=None,
        keepdim:bool=False,
        out:Union[Tensor,str]='')->Tensor:
    return _A_v_reduceop_C(a,dims,keepdim,"sum",out)

#prod
OpNode.register("prod")
def prod(
               a:Tensor,
        dims:Optional[Union[
            list[int],
            tuple[int],
            ]]=None,
        keepdim:bool=False,
        out:Union[Tensor,str]=''):
    return _A_v_reduceop_C(a,dims,keepdim,"prod",out)

#mean
OpNode.register("mean")
def mean(
        a:Tensor,
        dims:Optional[Union[list[int],tuple[int]]]=None,
        keepdim:bool=False,
        out:Union[str]=''):
    result=1/sum(a,dims,keepdim,out)
    return result
# #var
# OpNode.register("var")