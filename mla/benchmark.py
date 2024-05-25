from .configuration_deepseek import DeepseekV2Config
from .impl.baseline import DeepseekAttention as AttentionNoCache
from .impl.cache_decompressed import DeepseekAttention as AttentionCacheDecompressed
from .impl.cache_compressed import DeepseekAttention as AttentionCacheCompressed
from .impl.absorbed import DeepseekAttention as AttentionAbsorbed
from .impl.absorbed_cache_compressed import DeepseekAttention as AttentionAbsorbedCacheCompressed
import torch
import torch.utils.benchmark as benchmark

torch.set_grad_enabled(False)


class BenchmarkFixture:
    config: DeepseekV2Config
    q: torch.Tensor
    kv: torch.Tensor
    q_pos: torch.LongTensor
    kv_pos: torch.LongTensor

    def __init__(self, config: DeepseekV2Config, kv_len: int, q_len: int = 1, bsz: int = 1, dev='cuda'):
        self.config = config
        self.bsz = bsz
        self.q_len = q_len
        self.kv_len = kv_len
        self.q = torch.randn((self.bsz, self.q_len, config.hidden_size), dtype=config.torch_dtype, device=dev)
        self.kv = torch.randn((self.bsz, kv_len, config.hidden_size), dtype=config.torch_dtype, device=dev)
        self.q_pos = torch.randint(0, config.max_position_embeddings-1, (self.bsz, self.q_len), dtype=torch.long, device=dev)
        self.kv_pos = torch.arange(0, self.kv_len, dtype=torch.long, device=dev).unsqueeze(0).repeat(self.bsz, 1)
        cfg_dict = config.to_dict()
        cfg_dict['torch_dtype'] = config.torch_dtype
        self.cfg_dict = cfg_dict

    def benchmark(self, min_run_time: float = 1.0):
        return benchmark.Timer(
            stmt='bencher.iter()',
            globals={'bencher': self},
            label=self.name(),
            sub_label=f'kv_len={self.kv_len}',
        ).blocked_autorange(min_run_time=min_run_time)
    
    def name(self):
        return type(self).__name__.removesuffix('Bencher')

class BaselineBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, kv_len: int):
        super().__init__(config, kv_len)
        self.attn = AttentionNoCache(**self.cfg_dict).cuda()
    
    def iter(self):
        return self.attn.forward(self.q, self.kv, self.q_pos, self.kv_pos)

class CacheDecompressedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, kv_len: int):
        super().__init__(config, kv_len)
        self.attn = AttentionCacheDecompressed(**self.cfg_dict).cuda()
        self.decompressed = self.attn.decompress_kv(self.kv, self.kv_pos)
    
    def iter(self):
        k, v = self.decompressed
        return self.attn.forward(self.q, self.q_pos, k, v)

class CacheCompressedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, kv_len: int):
        super().__init__(config, kv_len)
        self.attn = AttentionCacheCompressed(**self.cfg_dict).cuda()
        self.compressed = self.attn.compress_kv(self.kv, self.kv_pos)
    
    def iter(self):
        return self.attn.forward(self.q, self.q_pos, self.compressed)

class AbsorbedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, kv_len: int):
        super().__init__(config, kv_len)
        self.attn = AttentionAbsorbed(**self.cfg_dict).cuda()
    
    def iter(self):
        return self.attn.forward(self.q, self.kv, self.q_pos, self.kv_pos)
    
class AbsorbedCacheCompressedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, kv_len: int):
        super().__init__(config, kv_len)
        self.attn = AttentionAbsorbedCacheCompressed(**self.cfg_dict).cuda()
        self.compressed = self.attn.compress_kv(self.kv, self.kv_pos)
    
    def iter(self):
        return self.attn.forward(self.q, self.q_pos, self.compressed)

ALL_BENCHMARKS = {
    'Baseline': BaselineBencher,
    'CacheDecompressed': CacheDecompressedBencher,
    'CacheCompressed': CacheCompressedBencher,
    'Absorbed': AbsorbedBencher,
    'AbsorbedCacheCompressed': AbsorbedCacheCompressedBencher
}

def main(bench: str, kv_len: int, config: str = 'mla/config.json', min_run_time: float=1.0):
    cfg = DeepseekV2Config.from_json_file(config)
    bencher: BenchmarkFixture
    bencher = ALL_BENCHMARKS[bench](cfg, kv_len)
    result = bencher.benchmark(min_run_time=min_run_time)
    del bencher
    print(result)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
    