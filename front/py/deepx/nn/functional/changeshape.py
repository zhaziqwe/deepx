from typing import Union,Tuple
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


def broadcast_shape(shape_a: tuple, shape_b: tuple) -> tuple:
    """计算两个形状的广播后形状"""
    # 获取形状的长度
    len_a, len_b = len(shape_a), len(shape_b)
    
    # 创建结果形状
    result_shape = []
    
    # 从右往左对齐并计算每个维度
    for i in range(1, min(len_a, len_b) + 1):
        dim_a = shape_a[-i]
        dim_b = shape_b[-i]
        
        if dim_a == 1 or dim_b == 1:
            # 广播规则：如果一个维度为1，取另一个维度的值
            result_shape.insert(0, max(dim_a, dim_b))
        elif dim_a == dim_b:
            # 维度相同，保持不变
            result_shape.insert(0, dim_a)
        else:
            # 维度不同且都不为1，无法广播
            raise ValueError(f"无法广播的形状：{shape_a} 和 {shape_b}")
    
    # 添加较长形状中多出的前导维度
    if len_a > len_b:
        result_shape = list(shape_a[:len_a - len_b]) + result_shape
    elif len_b > len_a:
        result_shape = list(shape_b[:len_b - len_a]) + result_shape
    
    return tuple(result_shape)


def unsqueeze(t:Tensor,dim:int)->Tensor:
    # 确保dim是有效的
    if dim < -t.ndim-1 or dim > t.ndim:
        raise ValueError(f"维度超出范围，当前张量维度为{t.ndim},dim={dim}")
    
    # 处理负数索引
    if dim < 0:
        dim = t.ndim + dim + 1

    new_shape = list(t.shape)
    new_shape.insert(dim, 1)

    return reshape(t, new_shape)

OpNode.register("expand")
def expand(t:Tensor,shape:list[int],out:Union[Tensor,str]='')->Tensor:
    outtensor=None
    if isinstance(out,str):
        outtensor=Tensor(shape=shape, dtype=t.dtype, device=t.device)
        outtensor.addtograph(out)
    else:
        outtensor=out

    opnode=t.graph.add_op("expand")
    opnode.add_input(t.node)
    opnode.add_input(t.graph.add_vector("",shape))
    outtensor.node.add_input(opnode)
    if t.graph.eager:
        ir=DeepxIR("expand",'',[t.node.name,*map(str, shape)], [outtensor.node.name])
        send(ir)
    return outtensor
# 修复 broadcast 函数缩进
def broadcast(a: Tensor, b: Tensor) -> Tuple[Tensor,Tensor]:
    # 计算广播后的形状
    try:
        target_shape = broadcast_shape(a.shape, b.shape)
    except ValueError as e:
        raise ValueError(f"广播失败：{e}") from e
    
    # 为每个张量添加前导维度
    if a.shape != target_shape:
        a_reshape = [1] * (len(target_shape) - a.ndimension) + list(a.shape)
        a_reshaped =  reshape(a,a_reshape)
    else:
        a_reshaped=a
    if b.shape != target_shape:
        b_reshape = [1] * (len(target_shape) - b.ndimension) + list(b.shape)
        b_reshaped = reshape(b,b_reshape)
    else:
        b_reshaped=b
    
    # 执行实际广播
    if a_reshaped.shape != target_shape:
        a_broadcasted =  expand(a_reshaped,target_shape)
    else:
        a_broadcasted=a_reshaped
        
    if b_reshaped.shape != target_shape:
        b_broadcasted =  expand(b_reshaped,target_shape)
    else:
        b_broadcasted=b_reshaped
    
    return a_broadcasted, b_broadcasted