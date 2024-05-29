from .configuration_deepseek import DeepseekV2Config
from .impl import *
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
    q_pos: torch.LongTensor
    kv_pos: torch.LongTensor

    def __init__(self, config: DeepseekV2Config, kv_len: int):
        self.config = config
        self.bsz = 16
        self.q_len = 1
        self.kv_len = kv_len
        dev = 'cuda'
        self.q = torch.randn((self.bsz, self.q_len, config.hidden_size), dtype=config.torch_dtype, device=dev)
        self.kv = torch.randn((self.bsz, kv_len, config.hidden_size), dtype=config.torch_dtype, device=dev)
        self.q_pos = torch.randint(0, config.max_position_embeddings-1, (self.bsz, self.q_len), dtype=torch.long, device=dev)
        self.kv_pos = torch.arange(0, self.kv_len, dtype=torch.long, device=dev).unsqueeze(0).repeat(self.bsz, 1)

fixture = Fixture(cfg, 1024)

def compute_error(std: torch.Tensor, x: torch.Tensor):
    '''
    Returns the relative L2 and Lmax between two tensors
    '''
    rl2e = linalg.vector_norm(std - x) / linalg.vector_norm(std)
    rlinfe = linalg.vector_norm(std - x, ord=math.inf) / linalg.vector_norm(std, ord=math.inf)
    return rl2e, rlinfe

baseline_attn = AttentionBaseline(**cfg_dict).cuda()
state_dict = baseline_attn.state_dict()
std_result = baseline_attn(fixture.q, fixture.kv, fixture.q_pos, fixture.kv_pos)

cache_decomporessed = AttentionCacheDecompressed(**cfg_dict).cuda()
cache_decomporessed.load_state_dict(state_dict)
k, v = cache_decomporessed.decompress_kv(fixture.kv, fixture.kv_pos)
result = cache_decomporessed(fixture.q, fixture.q_pos, k, v)
l2e, linfe = compute_error(std_result, result)
print(f'CacheDecomporessed: Relative L2 error={l2e}, Relative Linf error={linfe}')

cache_compressed = AttentionCacheCompressed(**cfg_dict).cuda()
cache_compressed.load_state_dict(state_dict)
compressed = cache_compressed.compress_kv(fixture.kv, fixture.kv_pos)
result = cache_compressed(fixture.q, fixture.q_pos, compressed)
l2e, linfe = compute_error(std_result, result)
print(f'CacheComporessed: Relative L2 error={l2e}, Relative Linf error={linfe}')

absorbed = AttentionAbsorbed(**cfg_dict).cuda()
absorbed.load_state_dict(state_dict)
result = absorbed(fixture.q, fixture.kv, fixture.q_pos, fixture.kv_pos)
l2e, linfe = compute_error(std_result, result)
print(f'Absorbed: Relative L2 error={l2e}, Relative Linf error={linfe}')

absorbed_cache_compressed = AttentionAbsorbed_CacheCompressed(**cfg_dict).cuda()
absorbed_cache_compressed.load_state_dict(state_dict)
compressed = absorbed_cache_compressed.compress_kv(fixture.kv, fixture.kv_pos)
result = absorbed_cache_compressed(fixture.q, fixture.q_pos, compressed)
l2e, linfe = compute_error(std_result, result)
print(f'Absorbed_CacheCompressed: Relative L2 error={l2e}, Relative Linf error={linfe}')

absorbed_cache_compressed_move_elision = AttentionAbsorbed_CacheCompressed_MoveElision(**cfg_dict).cuda()
absorbed_cache_compressed_move_elision.load_state_dict(state_dict)
compressed = absorbed_cache_compressed_move_elision.compress_kv(fixture.kv, fixture.kv_pos)
result = absorbed_cache_compressed_move_elision(fixture.q, fixture.q_pos, compressed)
l2e, linfe = compute_error(std_result, result)
print(f'Absorbed_CacheCompressed_MoveElision: Relative L2 error={l2e}, Relative Linf error={linfe}')

absorbed_materialized_cache_compressed_move_elision = AttentionAbsorbedMaterialized_CacheCompressed_MoveElision(**cfg_dict).cuda()
absorbed_materialized_cache_compressed_move_elision.load_state_dict(state_dict)
compressed = absorbed_materialized_cache_compressed_move_elision.compress_kv(fixture.kv, fixture.kv_pos)
result = absorbed_materialized_cache_compressed_move_elision(fixture.q, fixture.q_pos, compressed)
l2e, linfe = compute_error(std_result, result)
print(f'AbsorbedMaterialized_CacheCompressed_MoveElision: Relative L2 error={l2e}, Relative Linf error={linfe}')
