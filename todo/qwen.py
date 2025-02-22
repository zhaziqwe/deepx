# cursor 根据deepseek-qwen2 1.5b到处的deepx模型目录名，反推导出的pytorch模型
# DeepSeek-R1-Distill-Qwen-1.5B
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

class Qwen2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size  # 1536
        self.num_heads = config.num_attention_heads  # 12
        self.head_dim = self.hidden_size // self.num_heads  # 128
        self.num_key_value_heads = config.num_key_value_heads  # 2
        
        # 根据shape文件中的维度定义
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        # 实现GQA分组查询注意力逻辑
        # 包含RoPE位置编码实现（根据config.use_mrope决定）
        # 返回注意力计算结果
        return hidden_states

class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size  # 1536
        self.intermediate_size = config.intermediate_size  # 8960
        
        # 根据目录结构中的mlp层定义
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config)
        
        # 根据目录结构中的layernorm定义
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        
        self.mlp = Qwen2MLP(config)

    def forward(self, hidden_states, attention_mask=None):
        # 实现残差连接
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]  # 28层
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class Qwen2ForCausalLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 根据config.yaml中的参数设置
        self.config.tie_word_embeddings = False  # 不共享embedding权重

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
        return ModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )