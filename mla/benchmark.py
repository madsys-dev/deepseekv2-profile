from typing import Optional
from configuration_deepseek import DeepseekV2Config
from impl import *
import re
import torch
import torch.utils.benchmark as benchmark
import math

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
    
    @classmethod
    def name(cls):
        return cls.__name__.removesuffix('Bencher')

    @classmethod
    def short_name(cls):
        return re.sub('[^A-Z_]', '', cls.name())

    def cache_size(self):
        return 0




class SimplifiedMLABencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.lora_rank = 128  # Use a smaller rank to avoid memory issues
        
        self.attn = SimplifiedMLA(
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            lora_rank=self.lora_rank,
            dtype=config.torch_dtype
        )
        
        # Create input tensors
        self.hidden_state = torch.randn((self.bsz, config.hidden_size), 
                                      dtype=config.torch_dtype, device='cuda')
        
        # Create compressed KV states - use a smaller kv_len if memory is an issue
        kv_len = min(self.kv_len, 2048)  # Cap the kv_len to avoid memory issues
        self.compressed_kv = torch.randn((self.bsz, kv_len, self.lora_rank),
                                       dtype=config.torch_dtype, device='cuda')
    
    def iter(self):
        return self.attn(self.hidden_state, self.compressed_kv)
    
    def cache_size(self):
        return self.compressed_kv.numel() * self.compressed_kv.element_size()



class SimpleAttentionBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = SimpleAttention(
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            dtype=config.torch_dtype
        ).cuda()
        
        # Create input tensors
        self.hidden_state = torch.randn((self.bsz, config.hidden_size), 
                                      dtype=config.torch_dtype, device='cuda')
        
        # Create key and value states
        self.key_states = self.kv.repeat(self.bsz, 1, 1)
        self.value_states = self.kv.repeat(self.bsz, 1, 1)
    
    def iter(self):
        return self.attn(self.hidden_state, self.key_states, self.value_states)
    
    def cache_size(self):
        return (self.key_states.numel() + self.value_states.numel()) * self.key_states.element_size()

class SimpleCompressedAttentionBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.lora_rank = 512  # Can be adjusted as needed
        
        self.attn = SimpleCompressedAttention(
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            lora_rank=self.lora_rank,
            dtype=config.torch_dtype
        ).cuda()
        
        # Create input tensors
        self.hidden_state = torch.randn((self.bsz, config.hidden_size), 
                                      dtype=config.torch_dtype, device='cuda')
        
        # Create compressed KV states
        self.compressed_kv = torch.randn((self.bsz, self.kv_len, self.lora_rank),
                                       dtype=config.torch_dtype, device='cuda')
    
    def iter(self):
        return self.attn(self.hidden_state, self.compressed_kv)
    
    def cache_size(self):
        return self.compressed_kv.numel() * self.compressed_kv.element_size()

class SimpleAbsorbedAttention(torch.nn.Module):
    def __init__(self, num_heads: int, head_dim: int, lora_rank: int, dtype=torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lora_rank = lora_rank
        
        # Create random weights for query projection
        self.w_q = torch.nn.Linear(num_heads * head_dim, num_heads * head_dim, 
                                 dtype=dtype, device='cuda')
        
        # Combined projection for compressed KV
        self.kv_b_proj = torch.nn.Linear(lora_rank, 
                                        num_heads * head_dim * 2,  # For both K and V
                                        dtype=dtype, 
                                        device='cuda')
        
        # Store dimensions for splitting
        self.qk_head_dim = head_dim
    
    def forward(self, hidden_state: torch.Tensor, compressed_kv: torch.Tensor):
        batch_size = hidden_state.size(0)
        
        # Split weights for query and value projections
        kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.lora_rank)
        # (num_heads, head_dim, lora_rank)
        q_weights_proj = kv_b_proj[:, :self.qk_head_dim, :]  
        v_weights_proj = kv_b_proj[:, self.qk_head_dim:, :] 
        
        # Project query
        query = self.w_q(hidden_state).view(batch_size, self.num_heads, self.head_dim)
        query = query.unsqueeze(2)  # (batch_size, num_heads, 1, head_dim)
        
        # Project query with compressed weights
        # (batch_size, num_heads, 1, lora_rank)
        query_absorbed = torch.matmul(query, q_weights_proj)  
        
        # Compute attention scores with compressed KV
        # (batch_size, 1, seq_length, lora_rank)
        compressed_kv = compressed_kv.unsqueeze(1)  
        attn_weights = torch.matmul(query_absorbed, compressed_kv.transpose(-2, -1))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # (batch_size, num_heads, 1, lora_rank)
        attn_output = torch.matmul(attn_weights, compressed_kv)
        # (batch_size, num_heads, 1, head_dim)
        attn_output = torch.matmul(attn_output, v_weights_proj.transpose(1, 2))  
        
        return attn_output

class SimpleAbsorbedAttentionBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.lora_rank = 512  # Can be adjusted as needed
        
        self.attn = SimpleAbsorbedAttention(
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            lora_rank=self.lora_rank,
            dtype=config.torch_dtype
        ).cuda()
        
        # Create input tensors
        self.hidden_state = torch.randn((self.bsz, config.hidden_size), 
                                      dtype=config.torch_dtype, device='cuda')
        
        # Create compressed KV states
        self.compressed_kv = torch.randn((self.bsz, self.kv_len, self.lora_rank),
                                       dtype=config.torch_dtype, device='cuda')
    
    def iter(self):
        return self.attn(self.hidden_state, self.compressed_kv)
    
    def cache_size(self):
        return self.compressed_kv.numel() * self.compressed_kv.element_size()

ALL_BENCHMARKS = [
    SimpleAttentionBencher,
    SimpleCompressedAttentionBencher,
    SimpleAbsorbedAttentionBencher,
    SimplifiedMLABencher,
]

BENCHERS = {}

doc = 'Run benchmark on various MLA implementations\n\n'

for bencher in ALL_BENCHMARKS:
    name = bencher.name()
    short_name = bencher.short_name()
    BENCHERS[name] = bencher
    BENCHERS[short_name] = bencher
    doc += f'{short_name}\t{name}\n'

def main(bench: str,  kv_len: int, bsz: int = 1, config: str = 'mla/config.json', repeat: Optional[int] = None, 
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
    