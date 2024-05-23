from .configuration_deepseek import DeepseekV2Config
from .impl.baseline import DeepseekAttention as AttentionNoCache
from .impl.cache_decompressed import DeepseekAttention as AttentionCacheDecompressed
import torch

torch.set_grad_enabled(False)

cfg = DeepseekV2Config.from_json_file('baseline/config.json')

cfg_dict = cfg.to_dict()
cfg_dict['torch_dtype'] = cfg.torch_dtype


class BenchmarkFixture:
    config: DeepseekV2Config
    q: torch.Tensor
    kv: torch.Tensor
    q_pos: torch.LongTensor
    kv_pos: torch.LongTensor

    def __init__(self, config: DeepseekV2Config, kv_len: int):
        self.config = config
        self.bsz = 1
        self.q_len = 1
        self.kv_len = kv_len
        dev = 'cuda'
        self.q = torch.randn((self.bsz, self.q_len, config.hidden_size), dtype=config.torch_dtype, device=dev)
        self.kv = torch.randn((self.bsz, kv_len, config.hidden_size), dtype=config.torch_dtype, device=dev)
        self.q_pos = torch.randint(0, config.max_position_embeddings-1, (self.bsz, self.q_len), dtype=torch.long, device=dev)
        self.kv_pos = torch.arange(0, self.kv_len, dtype=torch.long, device=dev).unsqueeze(0).repeat(self.bsz, 1)


fixture = BenchmarkFixture(cfg, 1024)

baseline_attn = AttentionNoCache(**cfg_dict).cuda()
state_dict = baseline_attn.state_dict()
result = baseline_attn(fixture.q, fixture.kv, fixture.q_pos, fixture.kv_pos)
print(result)

cache_decomporessed = AttentionCacheDecompressed(**cfg_dict).cuda()
cache_decomporessed.load_state_dict(state_dict)
k, v = cache_decomporessed.decompress_kv(fixture.kv, fixture.kv_pos)
result = cache_decomporessed(fixture.q, fixture.q_pos, k, v)
print(result)

