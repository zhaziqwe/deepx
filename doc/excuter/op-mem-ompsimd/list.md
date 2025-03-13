## excuter/op-mem-ompsimd 支持算子列表 

本页面由 `excuter/op-mem-ompsimd/src/deepx/tf/tffactory.hpp` 生成，请勿手动修改 

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| argset |  none  | (arg)->(double) | shape = [3  4  5] | argset(arg )->(double d1) |
| argset |  none  | (arg)->(float) | shape = [3  4  5] | argset(arg )->(float f1) |
| argset |  none  | (args)->(int32) | shape = [3  4  5] | argset(args )->(int32 shape) |
| newtensor |  none  | (shape)->(double) | T1 = zeros(shape) | newtensor(shape )->(double tensor) |
| newtensor |  none  | (shape)->(float) | T1 = zeros(shape) | newtensor(shape )->(float tensor) |
| newtensor |  none  | (shape)->(int64) | T1 = zeros(shape) | newtensor(shape )->(int64 tensor) |
| newtensor |  none  | (shape)->(int32) | T1 = zeros(shape) | newtensor(shape )->(int32 tensor) |
| newtensor |  none  | (shape)->(int16) | T1 = zeros(shape) | newtensor(shape )->(int16 tensor) |
| newtensor |  none  | (shape)->(int8) | T1 = zeros(shape) | newtensor(shape )->(int8 tensor) |
