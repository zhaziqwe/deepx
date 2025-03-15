## excuter/op-mem-ompsimd 支持算子列表 

本页面由 `excuter/op-mem-ompsimd/src/deepx/tf/tffactory.hpp` 生成，请勿手动修改 

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| concat |  none  | (unknown<any>, var<int32>)->(tensor<any>) | Tresult = concat([T1, T2...], axis=3) | concat(unknown<any> tensors, var<int32> axis)->(tensor<any> Tresult) |
| newtensor |  none  | (var<unknown>)->(tensor<any>) | T1 = zeros(shape) | newtensor(var<unknown> shape)->(tensor<any> tensor1) |
| newtensor |  none  | (vector<int32>)->(tensor<any>) | T1 = zeros(shape) | newtensor(vector<int32> shape)->(tensor<any> tensor1) |
| vecset |  none  | (vector<any>)->() | shape = [3  4  5] | vecset(vector<any> shape)->() |
| argset |  none  | (var<any>)->() | var argname = argvalue | argset(var<any> argname)->() |
