from configuration_deepseek import DeepseekV2Config
from benchmark import SimpleAttention, SimpleCompressedAttention, SimpleAbsorbedAttention, SimplifiedMLA
import torch
from torch import linalg
import math

torch.set_grad_enabled(False)

cfg = DeepseekV2Config.from_json_file('mla/config.json')

cfg_dict = cfg.to_dict()
cfg_dict['torch_dtype'] = cfg.torch_dtype

class Fixture:
    config: DeepseekV2Config
    q: torch.Tensor
    kv: torch.Tensor

    def __init__(self, config: DeepseekV2Config, kv_len: int):
        self.config = config
        self.bsz = 16
        self.kv_len = kv_len
        dev = 'cuda'
        self.q = torch.randn((self.bsz, config.hidden_size), dtype=config.torch_dtype, device=dev)
        self.kv = torch.randn((self.bsz, kv_len, config.hidden_size), dtype=config.torch_dtype, device=dev)
        self.lora_rank = 512  # Rank for compressed models

fixture = Fixture(cfg, 1024)

def compute_error(std: torch.Tensor, x: torch.Tensor):
    '''
    Returns the relative L2 and Lmax between two tensors
    '''
    rl2e = linalg.vector_norm(std - x) / linalg.vector_norm(std)
    rlinfe = linalg.vector_norm(std - x, ord=math.inf) / linalg.vector_norm(std, ord=math.inf)
    return rl2e, rlinfe

# Create models
num_heads = cfg.num_attention_heads
head_dim = cfg.hidden_size // num_heads
dtype = cfg.torch_dtype

# 1. SimpleAttention (baseline)
simple_attn = SimpleAttention(num_heads=num_heads, head_dim=head_dim, dtype=dtype).cuda()
key_states = fixture.kv.reshape(fixture.bsz, -1, num_heads, head_dim).transpose(1, 2)
value_states = fixture.kv.reshape(fixture.bsz, -1, num_heads, head_dim).transpose(1, 2)
std_result = simple_attn(fixture.q, key_states, value_states)
# Reshape to consistent format for comparison: [batch_size, num_heads, head_dim]
std_result = std_result.squeeze(2)  # Remove sequence dimension
print(f"Baseline model: SimpleAttention")

# 2. SimpleCompressedAttention
simple_compressed = SimpleCompressedAttention(num_heads=num_heads, head_dim=head_dim, lora_rank=fixture.lora_rank, dtype=dtype).cuda()
# Create compressed KV
compressed_kv = torch.randn((fixture.bsz, fixture.kv_len, fixture.lora_rank), dtype=dtype, device='cuda')
result = simple_compressed(fixture.q, compressed_kv)
# Reshape to consistent format
result = result.squeeze(2)  # Remove sequence dimension
l2e, linfe = compute_error(std_result, result)
print(f'SimpleCompressedAttention: Relative L2 error={l2e}, Relative Linf error={linfe}')

# # 3. SimpleAbsorbedAttention
# simple_absorbed = SimpleAbsorbedAttention(num_heads=num_heads, head_dim=head_dim, lora_rank=fixture.lora_rank, dtype=dtype).cuda()
# result = simple_absorbed(fixture.q, compressed_kv)
# # Reshape to consistent format
# result = result.squeeze(2)  # Remove sequence dimension
# l2e, linfe = compute_error(std_result, result)
# print(f'SimpleAbsorbedAttention: Relative L2 error={l2e}, Relative Linf error={linfe}')

# # 4. SimplifiedMLA
# simplified_mla = SimplifiedMLA(num_heads=num_heads, head_dim=head_dim, lora_rank=fixture.lora_rank, dtype=dtype).cuda()
# result = simplified_mla(fixture.q, compressed_kv)
# # Reshape to consistent format
# result = result.squeeze(2)  # Remove sequence dimension
# l2e, linfe = compute_error(std_result, result)
# print(f'SimplifiedMLA: Relative L2 error={l2e}, Relative Linf error={linfe}')
