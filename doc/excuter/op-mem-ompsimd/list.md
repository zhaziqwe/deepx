## excuter/op-mem-ompsimd 支持算子列表

| Operation | Data Types | Math Formula | IR Instruction |
|-----------|------------|--------------|----------------|
| sum | float32, float64 | T2 = sum(T1, dims=[1,2]) | sum@float32 T1 1 2 -> T2 |
| matmul | float32, float64 | T3 = T1 @ T2 | matmul@float32 T1 T2 -> T3 |
| concat | float32, float64 | T3 = concat([T1, T2], axis=3) | concat@float32 T1 T2 3 -> T3 |
| max_scalar | float32, float64 | T2 = max(T1, 0.0) | max_scalar@float32 T1 0.0 -> T2 |
| exp | float32, float64 | T2 = exp(T1) | exp@float32 T1 -> T2 |
| min_scalar | float32, float64 | B= min(A, 1.0) | min_scalar@float32 A 1.0 -> B |
| sqrt | float32, float64 | T2 = sqrt(T1) | sqrt@float32 T1 -> T2 |
| div | float32, float64 | T3 = T1 / T2 | div@float32 T1 T2 -> T3 |
| mul | float32, float64 | T3 = T1 * T2 | mul@float32 T1 T2 -> T3 |
| newtensor | float32, float64, int16, int32, int64, int8 | T1 = zeros(shape) | newtensor@float32 shape -> T1 |
| print | any |  | print@any -> |
| min | float32, float64 | C = min(A,B) | min@float32 A B -> C |
| copytensor | float32, float64, int16, int32, int64, int8 | T2 = T1.copy() | copytensor@float32 T1 -> T2 |
| clonetensor | float32, float64, int16, int32, int64, int8 | T2 = T1.clone() | clonetensor@float32 T1 -> T2 |
| arange | float32, float64 | arange(start=0.0, step=1.0,T1) | arange@float32 0.0 1.0 -> T1 |
| argset | float32, float64, int32 | shape = [3, 4, 5] | argset@int32 3 4 5 -> shape |
| sub | float32, float64 | T3 = T1 - T2 | sub@int32 T1 T2 -> T3 |
| mul_scalar | float32, float64 | T2 = T1 * 2.0 | mul_scalar@float32 T1 2.0 -> T2 |
| uniform | float32, float64 | uniform(-1.0, 1.0,T1) | uniform@float32 -1.0 1.0 -> T1 |
| add | float32, float64 | T3 = T1 + T2 | add@int32 T1 T2 -> T3 |
| max | float32, float64 | T3 = max(T1,T2) | max@float32 T1 -> T2 |
| constant | float32, float64 | T1 = full(shape, 0.0) | constant@float32 0.0 -> T1 |
| rdiv_scalar | float32, float64 | T3 =1 / T2 | rdiv_scalar@float32 1 T2 -> T3 |
| add_scalar | float32, float64 | T2 = T1 + 1.0 | add_scalar@float32 T1 1.0 -> T2 |
| transpose | any | T2 = transpose(T1, dimorder=[1,0]) | transpose@float32 T1 1 0 -> T2 |
| div_scalar | float32, float64 | T2 = T1 / 2.0 | div_scalar@float32 T1 2.0 -> T2 |
| reshape | any | T2 = reshape(T1, [2,3,4]) | reshape@float32 T1 2 3 4 -> T2 |
