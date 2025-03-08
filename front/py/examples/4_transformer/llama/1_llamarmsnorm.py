
# 使用小规模数据以便打印完整结果
hidden_size = 8
eps = 1e-6


############### DeepX 实现部分 ###############
from deepx import arange, constant
from deepx.transformer.models.llama.modeling_llama import LlamaRMSNorm

# 使用相同的数据
dx_input = arange(0, 48, 1, dtype="float32").reshape_(2, 3, hidden_size)
dx_input.div_(10.0)
dx_input.sub_(2.0)
eps = 1e-6

print("\nDeepX 输入:")
print(dx_input)

# DeepX计算流程
dx_norm = LlamaRMSNorm(hidden_size=hidden_size, eps=eps)
# 设置相同的权重
constant(dx_norm.weight, 0.5)
# 前向计算
dx_output = dx_norm(dx_input)

print("\nDeepX RMSNorm 结果:")
print(dx_output)

import os
script_name = os.path.splitext(os.path.basename( os.path.abspath(__file__)))[0]  # 获取不带后缀的脚本名
str=dx_output.graph.to_dot()
str.render(script_name+".dot", format='svg')