from .elementwise import add,sub,mul,div,clamp
from .new import newtensor
from .print import printtensor
from .matmul import matmul
from .init import constant,full,zeros,ones,uniform_,arange,rand,randn,eye
from .reduce import max,min,sum,prod,mean
from .transpose import transpose
from .activite import relu

__all__ = [
    "newtensor",
    "printtensor",
    "constant","full","zeros","ones","uniform_","arange","rand","randn","eye",
    "add","sub","mul","div","clamp",
    "matmul",
    "max","min","sum","prod","mean",
    "transpose",    
    "relu",
]