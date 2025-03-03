from .elementwise import *
from .new import newtensor
from .print import printtensor
from .matmul import matmul
from .init import *
from .reduce import max,min,sum,prod,mean
from .transpose import transpose,reshape
from .activite import relu

__all__ = [
    "newtensor",
    "printtensor",
    "constant","full","zeros","ones","uniform","arange","rand","randn","eye","kaiming_uniform_",
    "add","sub","mul","div","clamp","exp","sqrt","rsqrt",
    "matmul",
    "max","min","sum","prod","mean",
    "transpose","reshape",
    "relu",
]