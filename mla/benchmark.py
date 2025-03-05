from typing import Optional
from configuration_deepseek import DeepseekV2Config
from impl import *
import re
import torch
import torch.utils.benchmark as benchmark
import math
from models import SimpleAttention, SimpleCompressedAttention, SimpleAbsorbedAttention


torch.set_grad_enabled(False)


class BenchmarkFixture:
    config: DeepseekV2Config

    def __init__(self, config: DeepseekV2Config, kv_len: int, q_len: int = 1, bsz: int = 1, dev='cuda'):
        self.config = config
        self.bsz = bsz
        self.kv_len = kv_len
        self.dev = dev
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
    
    @classmethod
    def name(cls):
        return cls.__name__.removesuffix('Bencher')

    @classmethod
    def short_name(cls):
        return re.sub('[^A-Z_]', '', cls.name())

    def cache_size(self):
        return 0


class SimpleAttentionBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        self.q_lora_rank = config.q_lora_rank if hasattr(config, 'q_lora_rank') else 1536
        
        self.attn = SimpleAttention(
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            dtype=config.torch_dtype,
            q_lora_rank=self.q_lora_rank,
        ).to(self.dev)
        
        # Create input tensors
        self.hidden_state = torch.randn(
            (self.bsz, self.q_lora_rank),
            dtype=config.torch_dtype, 
            device=self.dev
        )
        
        # Create key and value states
        head_dim = config.hidden_size // config.num_attention_heads
        self.key_states = torch.randn(
            (self.bsz, self.kv_len, config.num_attention_heads, head_dim),
            dtype=config.torch_dtype, 
            device=self.dev
        )
        self.value_states = torch.randn(
            (self.bsz, self.kv_len, config.num_attention_heads, head_dim),
            dtype=config.torch_dtype, 
            device=self.dev
        )
    
    def iter(self):
        return self.attn(self.hidden_state, self.key_states, self.value_states)
    
    def cache_size(self):
        return (self.key_states.numel() + self.value_states.numel()) * self.key_states.element_size()


class SimpleCompressedAttentionBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        self.kv_lora_rank = config.kv_lora_rank if hasattr(config, 'kv_lora_rank') else 512
        self.q_lora_rank = config.q_lora_rank if hasattr(config, 'q_lora_rank') else 1536
        
        self.attn = SimpleCompressedAttention(
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            dtype=config.torch_dtype,
        ).to(self.dev)
        
        # Create input tensors
        self.hidden_state = torch.randn(
            (self.bsz, self.q_lora_rank),
            dtype=config.torch_dtype, 
            device=self.dev
        )
        
        # Create compressed KV states
        self.compressed_kv = torch.randn(
            (self.bsz, self.kv_len, self.kv_lora_rank),
            dtype=config.torch_dtype, 
            device=self.dev
        )
    
    def iter(self):
        return self.attn(self.hidden_state, self.compressed_kv)
    
    def cache_size(self):
        return self.compressed_kv.numel() * self.compressed_kv.element_size()


class SimpleAbsorbedAttentionBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        self.kv_lora_rank = config.kv_lora_rank if hasattr(config, 'kv_lora_rank') else 512
        self.q_lora_rank = config.q_lora_rank if hasattr(config, 'q_lora_rank') else 1536
        
        self.attn = SimpleAbsorbedAttention(
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            dtype=config.torch_dtype
        ).to(self.dev)
        
        # Create input tensors
        self.hidden_state = torch.randn(
            (self.bsz, self.q_lora_rank),
            dtype=config.torch_dtype, 
            device=self.dev
        )
        
        # Create compressed KV states
        self.compressed_kv = torch.randn(
            (self.bsz, self.kv_len, self.kv_lora_rank),
            dtype=config.torch_dtype, 
            device=self.dev
        )
    
    def iter(self):
        return self.attn(self.hidden_state, self.compressed_kv)
    
    def cache_size(self):
        return self.compressed_kv.numel() * self.compressed_kv.element_size()


ALL_BENCHMARKS = [
    SimpleAttentionBencher,
    SimpleCompressedAttentionBencher,
    SimpleAbsorbedAttentionBencher,
]

BENCHERS = {}

doc = 'Run benchmark on various MLA implementations\n\n'

for bencher in ALL_BENCHMARKS:
    name = bencher.name()
    short_name = bencher.short_name()
    BENCHERS[name] = bencher
    BENCHERS[short_name] = bencher
    doc += f'{short_name}\t{name}\n'


def main(bench: str, kv_len: int, bsz: int = 1, config: str = 'mla/config.json', repeat: Optional[int] = None, 
         min_run_time: float = 1.0, csv: bool = False):
    cfg = DeepseekV2Config.from_json_file(config)
    bencher: BenchmarkFixture
    bencher = BENCHERS[bench](cfg, kv_len, bsz=bsz)
    if repeat is not None:
        for _ in range(repeat):
            bencher.iter()
        torch.cuda.synchronize()
        return
    result = bencher.benchmark(min_run_time=min_run_time)
    cache_size = bencher.cache_size()
    device_name = torch.cuda.get_device_name()
    
    # Print results in key: value format
    print(f"Model: {bencher.name()}")
    print(f"Batch_Size: {bsz}")
    print(f"KV_Length: {kv_len}")
    print(f"Device: {device_name}")
    print(f"Cache_Size: {cache_size}")
    print(f"Mean: {result.mean}")
    print(f"Median: {result.median}")
    print(f"P25: {result._p25}")
    print(f"P75: {result._p75}")


main.__doc__ = doc

if __name__ == "__main__":
    import fire
    fire.Fire(main)
    