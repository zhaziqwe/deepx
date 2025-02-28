from .elementwise import add,sub,mul,div,clamp
from .new import newtensor
from .print import printtensor
from .matmul import matmul
from .init import full,zeros,ones,arange,rand,randn,eye
from .reduce import max,min,sum,prod,mean

__all__ = [
    "newtensor",
    "printtensor",
    "full","zeros","ones","arange","rand","randn","eye",
    "add","sub","mul","div","clamp",
    "matmul",
    "max","min","sum","prod","mean",
]