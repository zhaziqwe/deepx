## excuter/op-mem-ompsimd 支持算子列表 

本页面由 `excuter/op-mem-ompsimd/src/deepx/tf/tffactory.hpp` 生成，请勿手动修改 

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| concat |  none  | concat(listtensor<any> tensors, var<int32> axis)->(tensor<any> Tresult) | Tresult = concat([T1, T2...], axis=3) | concat(listtensor<any> tensors, var<int32> axis)->(tensor<any> Tresult) |
| print |  none  | print(tensor<any> tensor1, var<string> format)->() | print(T1) | print(tensor<any> tensor1, var<string> format)->() |
| print |  none  | print(tensor<any> tensor1)->() | print(T1) | print(tensor<any> tensor1)->() |
| newtensor |  none  | newtensor(vector<int32> shape)->(tensor<any> tensor1) | T1 = zeros(shape) | newtensor(vector<int32> shape)->(tensor<any> tensor1) |
| newtensor |  none  | newtensor(var<string> shape)->(tensor<any> tensor1) | T1 = zeros(shape) | newtensor(var<string> shape)->(tensor<any> tensor1) |
| vecset |  none  | vecset(vector<any> shape)->(vector<any> name) | shape = [3  4  5] | vecset(vector<any> shape)->(vector<any> name) |
| argset |  none  | argset(var<any> value)->(var<any> name) | var argname = argvalue | argset(var<any> value)->(var<any> name) |
