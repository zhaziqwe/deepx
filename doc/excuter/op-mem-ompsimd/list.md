## excuter/op-mem-ompsimd 支持算子列表 

本页面由 `excuter/op-mem-ompsimd/src/deepx/op/opfactory.hpp` 生成，请勿手动修改 

| Operation | Author | Data Types | Math Formula | IR Instruction |
|-----------|--------|------------|--------------|----------------|
| divscalar | miaobyte | float32, float64 | T2 = T1 / 2.0 | divscalar@float32 T1 2.0 -> T2 |
| addscalar | miaobyte | float32, float64 | T2 = T1 + 1.0 | addscalar@float32 T1 1.0 -> T2 |
| uniform |  | float32, float64 | uniform(-1.0, 1.0,T1) | uniform@float32 -1.0 1.0 -> T1 |
| deltensor |  | any | del T1 | deltensor@any T1 -> |
| minscalar |  | float32, float64 | B= min(A, 1.0) | minscalar@float32 A 1.0 -> B |
| rdivscalar | miaobyte | float32, float64 | T3 =1 / T2 | rdivscalar@float32 1 T2 -> T3 |
| constant |  | float32, float64 | T1 = full(shape, 0.0) | constant@float32 0.0 -> T1 |
| powscalar | miaobyte | float32, float64 | T2 = T1 ^ 2.0 | powscalar@float32 T1 2.0 -> T2 |
| sub | cblas | float32, float64 | T3 = T1 - T2 | sub@int32 T1 T2 -> T3 |
| sub | miaobyte | float32, float64 | T3 = T1 - T2 | sub@int32 T1 T2 -> T3 |
| sum |  | float32, float64 | T2 = sum(T1, dims=[1,2]) | sum@float32 T1 1 2 -> T2 |
| argset |  | float32, float64, int32 | shape = [3, 4, 5] | argset@int32 3 4 5 -> shape |
| arange |  | float32, float64 | arange(start=0.0, step=1.0,T1) | arange@float32 0.0 1.0 -> T1 |
| transpose |  | any | T2 = transpose(T1, dimorder=[1,0]) | transpose@float32 T1 1 0 -> T2 |
| clonetensor |  | float32, float64, int16, int32, int64, int8 | T2 = T1.clone() | clonetensor@float32 T1 -> T2 |
| add | cblas | float32, float64 | T3 = T1 + T2 | add@int32 T1 T2 -> T3 |
| add | miaobyte | float32, float64, int16, int32, int64, int8 | T3 = T1 + T2 | add@int32 T1 T2 -> T3 |
| copytensor |  | float32, float64, int16, int32, int64, int8 | T2 = T1.copy() | copytensor@float32 T1 -> T2 |
| min |  | float32, float64 | C = min(A,B) | min@float32 A B -> C |
| print |  | any |  | print@any -> |
| newtensor |  | float32, float64, int16, int32, int64, int8 | T1 = zeros(shape) | newtensor@float32 shape -> T1 |
| mulscalar | miaobyte | float32, float64 | T2 = T1 * 2.0 | mulscalar@float32 T1 2.0 -> T2 |
| div | miaobyte | float32, float64 | T3 = T1 / T2 | div_miaobyte@float32 T1 T2 -> T3 |
| sqrt | miaobyte | float32, float64 | T2 = sqrt(T1) | sqrt@float32 T1 -> T2 |
| mul | miaobyte | float32, float64 | T3 = T1 * T2 | mul@float32 T1 T2 -> T3 |
| exp | miaobyte | float32, float64 | T2 = exp(T1) | exp@float32 T1 -> T2 |
| max |  | float32, float64 | T3 = max(T1,T2) | max@float32 T1 -> T2 |
| pow | miaobyte | float32, float64 | T3 = T1 ^ T2 | pow@float32 T1 T2 -> T3 |
| maxscalar |  | float32, float64 | T2 = max(T1, 0.0) | maxscalar@float32 T1 0.0 -> T2 |
| matmul |  | float32, float64 | T3 = T1 @ T2 | matmul@float32 T1 T2 -> T3 |
| reshape |  | any | T2 = reshape(T1, [2,3,4]) | reshape@float32 T1 2 3 4 -> T2 |
| expand |  | any | T2 = expand(T1, axis=[4,6,12]) | expand@float32 T1 4 6 12 -> T2 |
| concat |  | float32 | T3 = concat([T1, T2], axis=3) | concat@float32 T1 T2 3 -> T3 |
