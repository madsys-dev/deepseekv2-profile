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


class SimplifiedMLA(torch.nn.Module):
    """
    Simplified Multi-Latent Attention with weight absorption trick.
    No RoPE, no quantization, no layernorm - just the core mechanism.
    """
    def __init__(self, num_heads: int, head_dim: int, lora_rank: int, dtype=torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim
        self.lora_rank = lora_rank
        
        # Query projection
        self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device='cuda')
        
        # KV latent projection matrix (down projection to latent space)
        self.kv_proj = torch.nn.Linear(self.hidden_size, self.lora_rank, dtype=dtype, device='cuda')
        
        # Absorbed weight matrices
        # w_kc: Key compression weight for query absorption (W_uk in the paper)
        # w_vc: Value compression weight for output projection (W_uv in the paper)
        self.w_kc = torch.nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.lora_rank, dtype=dtype, device='cuda')
        )
        self.w_vc = torch.nn.Parameter(
            torch.randn(self.num_heads, self.lora_rank, self.head_dim, dtype=dtype, device='cuda')
        )
        
        # Final output projection
        self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device='cuda')
        
        # Scaling factor for attention
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, hidden_states: torch.Tensor, compressed_kv: torch.Tensor = None):
        """
        Args:
            hidden_states: Input tensor [batch_size, hidden_size]
            compressed_kv: Pre-computed KV cache [batch_size, seq_len, lora_rank] or None
            
        Returns:
            output: Attention output [batch_size, hidden_size]
        """
        batch_size = hidden_states.size(0)
        
        # Create compressed KV if not provided
        if compressed_kv is None:
            # In real usage, this would be the KV cache from previous tokens
            # For benchmarking, we'll create a dummy sequence
            seq_len = 1024  # Example sequence length
            compressed_kv = self.kv_proj(
                torch.randn(batch_size, seq_len, self.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
            )
        else:
            seq_len = compressed_kv.size(1)
        
        # Project query and reshape
        q = self.q_proj(hidden_states).view(batch_size, self.num_heads, self.head_dim)
        
        # Absorb the key weights into the query (this is the key absorption trick)
        # q_absorbed shape: [batch_size, num_heads, lora_rank]
        # This avoids expanding the compressed KV cache
        q_absorbed = torch.bmm(
            q.view(batch_size * self.num_heads, 1, self.head_dim),
            self.w_kc.reshape(self.num_heads * self.head_dim, self.lora_rank).view(self.num_heads, self.head_dim, self.lora_rank)
        ).view(batch_size, self.num_heads, self.lora_rank)
        
        # Apply attention in the compressed latent space
        # Reshape compressed_kv to [batch_size, num_heads, seq_len, lora_rank]
        kv_latent = compressed_kv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Calculate attention scores: [batch_size, num_heads, seq_len]
        attn_scores = torch.matmul(
            q_absorbed.unsqueeze(2),  # [batch_size, num_heads, 1, lora_rank]
            kv_latent.transpose(-1, -2)  # [batch_size, num_heads, lora_rank, seq_len]
        ).squeeze(2) * self.scale
        
        # Apply softmax: [batch_size, num_heads, seq_len]
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to get the context vectors
        # [batch_size, num_heads, lora_rank]
        attn_output = torch.bmm(
            attn_weights.unsqueeze(2),  # [batch_size, num_heads, 1, seq_len]
            kv_latent  # [batch_size, num_heads, seq_len, lora_rank]
        ).squeeze(2)
        
        # Project back to head dimension using value weights
        # [batch_size, num_heads, head_dim]
        attn_output = torch.bmm(
            attn_output.view(batch_size * self.num_heads, 1, self.lora_rank),
            self.w_vc.reshape(self.num_heads, self.lora_rank, self.head_dim).repeat(batch_size, 1, 1)
        ).view(batch_size, self.num_heads, self.head_dim)
        
        # Reshape output and apply final projection
        attn_output = attn_output.reshape(batch_size, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output


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


class SimpleAttention(torch.nn.Module):
    def __init__(self, num_heads: int, head_dim: int, dtype=torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Create random weights for query projection
        self.w_q = torch.nn.Linear(num_heads * head_dim, num_heads * head_dim, 
                                 dtype=dtype, device='cuda')
    
    def forward(self, hidden_state: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor):
        batch_size = hidden_state.size(0)
        
        # Project current token to query space and split into heads
        query = self.w_q(hidden_state).view(batch_size, self.num_heads, self.head_dim)
        
        # Reshape key_states and value_states to split by heads
        key_states = key_states.view(batch_size, -1, self.num_heads, 
                                   self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_heads, 
                                       self.head_dim).transpose(1, 2)
        
        # Add sequence dimension to query for attention
        query = query.unsqueeze(2)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query, key_states.transpose(-2, -1))
        
        # Apply softmax
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Get attention output
        attention_output = torch.matmul(attention_probs, value_states)
        
        return attention_output

class SimpleCompressedAttention(torch.nn.Module):
    def __init__(self, num_heads: int, head_dim: int, lora_rank: int, dtype=torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lora_rank = lora_rank
        
        # Create random weights for projections
        self.w_q = torch.nn.Linear(num_heads * head_dim, num_heads * head_dim, 
                                 dtype=dtype, device='cuda')
        
        # Combined projection for compressed KV
        self.kv_b_proj = torch.nn.Linear(lora_rank, 
                                        num_heads * head_dim * 2,  # For both K and V, per head
                                        dtype=dtype, 
                                        device='cuda')
        
        # Scaling factor for attention
        self.softmax_scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, hidden_state: torch.Tensor, compressed_kv: torch.Tensor):
        batch_size = hidden_state.size(0)
        kv_seq_len = compressed_kv.size(1)
        
        # Project query and split into heads
        query = self.w_q(hidden_state) \
            .view(batch_size, self.num_heads, self.head_dim) \
            .unsqueeze(2)
        
        # Project compressed KV and split into heads
        kv = self.kv_b_proj(compressed_kv) \
            .view(batch_size, kv_seq_len, self.num_heads, 2, self.head_dim) \
            .permute(0, 2, 1, 3, 4) \
            .reshape(batch_size, self.num_heads, kv_seq_len, 2 * self.head_dim)
        
        # Split KV into key and value
        key, value = torch.split(kv, [self.head_dim, self.head_dim], dim=-1)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.softmax_scale
        
        # Apply softmax
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Get attention output
        attention_output = torch.matmul(attention_probs, value)
        
        return attention_output

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
    