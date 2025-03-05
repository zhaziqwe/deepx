from .elementwise import *
from .new import newtensor
from .print import printtensor
from .matmul import matmul
from .init import *
from .reduce import reduce_max,reduce_min,sum,prod,mean
from .transpose import transpose,reshape
from .activite import relu

__all__ = [
    "newtensor",
    "printtensor",
    "constant","full","zeros","ones","uniform","arange","rand","randn","eye","kaiming_uniform_","calculate_fan_in_and_fan_out",
    "add","sub","mul","div","clamp","exp","sqrt","rsqrt",
    "matmul",
    "max","min","sum","prod","mean",
    "transpose","reshape",
    "relu",
]