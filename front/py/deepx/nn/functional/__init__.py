from .elementwise import add,sub,mul,div
from .new import newtensor
from .print import printtensor
from .matmul import matmul
from .init import full,zeros,ones,arange,rand,randn,eye
__all__ = [
    "newtensor",
    "printtensor",
    "full","zeros","ones","arange","rand","randn","eye",
    "add","sub","mul","div",
    "matmul",
]