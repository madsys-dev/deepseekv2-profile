from typing import Optional
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
        self.kv = torch.randn((1, kv_len, config.hidden_size), dtype=config.torch_dtype, device=dev)
        self.q_pos = torch.randint(0, config.max_position_embeddings-1, (self.bsz, self.q_len), dtype=torch.long, device=dev)
        self.kv_pos = torch.arange(0, self.kv_len, dtype=torch.long, device=dev).unsqueeze(0)
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

    def cache_size(self):
        return 0    

class BaselineBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionNoCache(**self.cfg_dict).cuda()
        self.kv = self.kv.repeat(self.bsz, 1, 1)
        self.kv_pos = self.kv_pos.repeat(self.bsz, 1)
    
    def iter(self):
        return self.attn.forward(self.q, self.kv, self.q_pos, self.kv_pos)

class CacheDecompressedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionCacheDecompressed(**self.cfg_dict).cuda()
        k, v = self.attn.decompress_kv(self.kv, self.kv_pos)
        self.decompressed = k.repeat(self.bsz, 1, 1, 1), v.repeat(self.bsz, 1, 1, 1)

    def iter(self):
        k, v = self.decompressed
        return self.attn.forward(self.q, self.q_pos, k, v)
    
    def cache_size(self):
        k, v = self.decompressed
        return k.numel() * k.element_size() + v.numel() * v.element_size()

class CacheCompressedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionCacheCompressed(**self.cfg_dict).cuda()
        self.compressed = self.attn.compress_kv(self.kv, self.kv_pos).repeat(self.bsz, 1, 1)
    
    def iter(self):
        return self.attn.forward(self.q, self.q_pos, self.compressed)
    
    def cache_size(self):
        return self.compressed.numel() * self.compressed.element_size()

class AbsorbedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionAbsorbed(**self.cfg_dict).cuda()
        self.kv = self.kv.repeat(self.bsz, 1, 1)
        self.kv_pos = self.kv_pos.repeat(self.bsz, 1)
    
    def iter(self):
        return self.attn.forward(self.q, self.kv, self.q_pos, self.kv_pos)

class AbsorbedCacheCompressedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionAbsorbedCacheCompressed(**self.cfg_dict).cuda()
        self.compressed = self.attn.compress_kv(self.kv, self.kv_pos).repeat(self.bsz, 1, 1)

    def iter(self):
        return self.attn.forward(self.q, self.q_pos, self.compressed)
    
    def cache_size(self):
        return self.compressed.numel() * self.compressed.element_size()


ALL_BENCHMARKS = {
    'Baseline': BaselineBencher,
    'CacheDecompressed': CacheDecompressedBencher,
    'CacheCompressed': CacheCompressedBencher,
    'Absorbed': AbsorbedBencher,
    'AbsorbedCacheCompressed': AbsorbedCacheCompressedBencher
}

def main(bench: str,  kv_len: int, bsz: int = 1, config: str = 'mla/config.json', repeat: Optional[int] = None, 
         min_run_time: float = 1.0, csv: bool = False):
    cfg = DeepseekV2Config.from_json_file(config)
    bencher: BenchmarkFixture
    bencher = ALL_BENCHMARKS[bench](cfg, kv_len, bsz=bsz)
    if repeat is not None:
        for _ in range(repeat):
            bencher.iter()
        torch.cuda.synchronize()
        return
    result = bencher.benchmark(min_run_time=min_run_time)
    cache_size = bencher.cache_size()
    device_name = torch.cuda.get_device_name()
    del bencher
    if csv:
        print(f'{bench},{bsz},{kv_len},{device_name},{cache_size},{result.mean},{result.median},{result._p25},{result._p75}')
    else:
        print(result)
        print(f'Device: {device_name}')
        print(f'KV Cache: {cache_size}')

if __name__ == "__main__":
    import fire
    fire.Fire(main)
    