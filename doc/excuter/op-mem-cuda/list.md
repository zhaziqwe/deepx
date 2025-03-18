## op-mem-cuda 支持算子列表 

本页面由 `excuter/op-mem-cuda 生成，请勿手动修改 

| Operation | Author | Func Def | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| print |  none  | print(tensor<any> tensor1, var<string> format)->() | print(T1) | print(tensor<any> tensor1, var<string> format)->() |
| print |  none  | print(tensor<any> tensor1)->() | print(T1) | print(tensor<any> tensor1)->() |
| newtensor |  none  | newtensor(vector<int32> shape)->(tensor<any> tensor1) | T1 = zeros(shape) | newtensor(vector<int32> shape)->(tensor<any> tensor1) |
| newtensor |  none  | newtensor(var<string> shape)->(tensor<any> tensor1) | T1 = zeros(shape) | newtensor(var<string> shape)->(tensor<any> tensor1) |
| vecset |  none  | vecset(vector<any> shape)->(vector<any> name) | shape = [3  4  5] | vecset(vector<any> shape)->(vector<any> name) |
| argset |  none  | argset(var<any> value)->(var<any> name) | var argname = argvalue | argset(var<any> value)->(var<any> name) |
