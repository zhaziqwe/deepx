from enum import Enum, auto

class DType(Enum):
    COMPLEX64 = auto()
    COMPLEX128 = auto()
    BFLOAT16 = auto()
    FLOAT16 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    UINT8 = auto()
    INT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    INT32 = auto()
    UINT64 = auto()
    INT64 = auto()
    BOOL = auto()

def _dtype_to_typestr(dtype):
    return {
        DType.COMPLEX64: "<c8",
        DType.COMPLEX128: "<c16", 
        DType.BFLOAT16: "<V2",
        DType.FLOAT16: "<f2",
        DType.FLOAT32: "<f4",
        DType.FLOAT64: "<f8",
        DType.UINT8: "|u1",
        DType.INT8: "|i1",
        DType.UINT16: "<u2",
        DType.INT32: "<i4",
        DType.UINT32: "<u4",
        DType.UINT64: "<u8",
        DType.INT64: "<i8",
        DType.BOOL: "|b1",
    }[dtype] 