from typing import Optional
from configuration_deepseek import DeepseekV2Config
from impl import *
import re
import torch
import torch.utils.benchmark as benchmark
import math
import time

class SimpleAttention(torch.nn.Module):
    def __init__(self, num_heads: int, head_dim: int, q_lora_rank: int, dtype=torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.dtype = dtype
        
        # Add projection for hidden_states_q
        self.q_proj = torch.nn.Linear(q_lora_rank, num_heads * head_dim, dtype=dtype, device='cuda')
        
    def forward(self, hidden_states_q, key, value):
        """
        Input:
        - hidden_states_q: [bsz, hidden_size]
        - key: [bsz, seq_len, num_heads, head_dim]
        - value: [bsz, seq_len, num_heads, head_dim]
        """
        bsz = hidden_states_q.shape[0]
        
        # Project hidden_states_q
        q = self.q_proj(hidden_states_q)
        q = q.view(bsz, self.num_heads, self.head_dim)
        
        # Reshape key and value for attention computation
        seq_len = key.shape[1]
        key = key.transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
        value = value.transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        attn_weights = torch.matmul(q.unsqueeze(2), key.transpose(2, 3)).squeeze(2)  # [bsz, num_heads, seq_len]
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # Apply attention to value
        attn_output = torch.matmul(attn_weights.unsqueeze(2), value).squeeze(2)  # [bsz, num_heads, head_dim]
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim)
        
        return attn_output


class SimpleCompressedAttention(torch.nn.Module):
    def __init__(self, num_heads: int, head_dim: int, lora_rank: int, q_lora_rank: int, dtype=torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lora_rank = lora_rank
        self.q_lora_rank = q_lora_rank
        self.dtype = dtype
        
        # Add projection for hidden_states_q
        self.q_proj = torch.nn.Linear(q_lora_rank, num_heads * head_dim, dtype=dtype, device='cuda')
        self.w_kv = torch.nn.Linear(lora_rank, num_heads * head_dim, dtype=dtype, device='cuda')
        
    def forward(self, hidden_states_q, compressed_kv):
        """
        Input:
        - hidden_states_q: [bsz, hidden_size]
        - compressed_kv: [bsz, seq_len, lora_rank]
        """
        bsz = hidden_states_q.shape[0]
        seq_len = compressed_kv.shape[1]
        
        # Project hidden_states_q
        q = self.q_proj(hidden_states_q)
        q = q.view(bsz, self.num_heads, self.head_dim)
        
        # Project compressed_kv to key-value
        kv = self.w_kv(compressed_kv)  # [bsz, seq_len, num_heads * head_dim]
        kv = kv.view(bsz, seq_len, self.num_heads, self.head_dim)
        kv = kv.transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        attn_weights = torch.matmul(q.unsqueeze(2), kv.transpose(2, 3)).squeeze(2)  # [bsz, num_heads, seq_len]
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # Apply attention to value (using the same kv for simplicity)
        attn_output = torch.matmul(attn_weights.unsqueeze(2), kv).squeeze(2)  # [bsz, num_heads, head_dim]
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim)
        
        return attn_output


class SimpleAbsorbedAttention(torch.nn.Module):
    def __init__(self, num_heads: int, head_dim: int, lora_rank: int, q_lora_rank: int, dtype=torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lora_rank = lora_rank
        self.q_lora_rank = q_lora_rank
        self.dtype = dtype
        self.scaling = head_dim ** -0.5
        
        # Query projection
        self.q_proj = torch.nn.Linear(q_lora_rank, num_heads * head_dim, dtype=dtype, device='cuda')
        
        # Key-Value projection matrix
        self.kv_b_proj = torch.nn.Linear(lora_rank, 
                                        num_heads * head_dim * 2,  # For both K and V
                                        bias=False,
                                        dtype=dtype, 
                                        device='cuda')
        
        # Extract and store key and value projection matrices during initialization
        # Similar to how they do it in DeepSeekV2AttentionMLA
        kv_b_proj = self.kv_b_proj.weight
        self.w_kc, self.w_vc = kv_b_proj.unflatten(0, (self.num_heads, 2 * self.head_dim)).split(
            [self.head_dim, self.head_dim], dim=1
        )
        # Reshape for easier matrix multiplication in forward pass
        self.w_kc = self.w_kc.reshape(self.num_heads, self.head_dim, self.lora_rank)
        self.w_vc = self.w_vc.reshape(self.num_heads, self.head_dim, self.lora_rank)
    
    def forward(self, hidden_states_q: torch.Tensor, compressed_kv: torch.Tensor):
        """
        Input:
        - hidden_states_q: [bsz, q_lora_rank]
        - compressed_kv: [bsz, seq_len, lora_rank]
        """
        bsz = hidden_states_q.shape[0]
        seq_len = compressed_kv.shape[1]
        
        # Project query
        q = self.q_proj(hidden_states_q).view(bsz, self.num_heads, self.head_dim)
        
        # Weight absorption trick: project query with key weights
        # Use einsum for clearer matrix multiplication
        # q: [bsz, num_heads, head_dim]
        # self.w_kc: [num_heads, head_dim, lora_rank]
        # Result: [bsz, num_heads, lora_rank]
        q_absorbed = torch.einsum('bhd,hdr->bhr', q, self.w_kc)
        
        # Calculate attention scores with einsum
        # q_absorbed: [bsz, num_heads, lora_rank]
        # compressed_kv: [bsz, seq_len, lora_rank]
        # Result: [bsz, num_heads, seq_len]
        attn_scores = torch.einsum('bhr,bsr->bhs', q_absorbed, compressed_kv)
        
        # Apply softmax to get attention probabilities
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)  # [bsz, num_heads, seq_len]
        
        # Apply attention weights to compressed values
        # attn_probs: [bsz, num_heads, seq_len]
        # compressed_kv: [bsz, seq_len, lora_rank]
        # Result: [bsz, num_heads, lora_rank]
        context = torch.einsum('bhs,bsr->bhr', attn_probs, compressed_kv)
        
        # Project back using value weights with einsum
        # context: [bsz, num_heads, lora_rank]
        # self.w_vc: [num_heads, head_dim, lora_rank]
        # Result: [bsz, num_heads, head_dim]
        attn_output = torch.einsum('bhr,hdr->bhd', context, self.w_vc)
        
        # Reshape to final output
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim)
        
        return attn_output



# Add this at the bottom of the file
if __name__ == "__main__":
    print("\nRunning SimpleAttention test...")

    # Set CUDA device to GPU 5
    torch.cuda.set_device(5)
    print(f"Using CUDA device: {torch.cuda.current_device()}")

    torch.set_grad_enabled(False)
    # Create a simple test case
    test_batch_size = 2
    test_seq_len = 1000
    test_heads = 128#32
    head_dim = 5120
    test_lora_rank = 512
    q_lora_rank = 1536

    # Choose precision
    dtype = torch.float16  # or torch.bfloat16 or torch.float32
    
    # Create test inputs with specified precision
    test_hidden_q = torch.randn((test_batch_size, q_lora_rank), dtype=dtype, device='cuda')
    test_key = torch.randn((test_batch_size, test_seq_len, test_heads, head_dim), dtype=dtype, device='cuda')
    test_value = torch.randn((test_batch_size, test_seq_len, test_heads, head_dim), dtype=dtype, device='cuda')
    
    # Create model with specified precision
    test_simple_attn = SimpleAttention(num_heads=test_heads, head_dim=head_dim, dtype=dtype, q_lora_rank=q_lora_rank).cuda()
    test_simple_compressed_attn = SimpleCompressedAttention(num_heads=test_heads, head_dim=head_dim, lora_rank=test_lora_rank, q_lora_rank=q_lora_rank, dtype=dtype).cuda()
    test_simple_absorbed_attn = SimpleAbsorbedAttention(num_heads=test_heads, head_dim=head_dim, lora_rank=test_lora_rank, q_lora_rank=q_lora_rank, dtype=dtype).cuda()

    # Run forward pass with timing (average of 10 runs)
    print(f"Input shape: {test_hidden_q.shape}")
    print(f"Key shape: {test_key.shape}")
    print(f"Value shape: {test_value.shape}")
    
    # Burn-in runs
    burn_in = 2
    for _ in range(burn_in):
        _ = test_simple_attn(test_hidden_q, test_key, test_value)
        torch.cuda.synchronize()
    
    # Time over 10 runs
    num_runs = 10
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        output = test_simple_attn(test_hidden_q, test_key, test_value)
        torch.cuda.synchronize()
        total_time += time.time() - start_time
    
    avg_time = total_time / num_runs
    print(f"Output shape: {output.shape}")
    print(f"Time taken (avg of {num_runs} runs, after {burn_in} burn-in): {avg_time:.6f} seconds")

    # Run compressed attention with timing (average of 10 runs)
    test_compressed_kv = torch.randn((test_batch_size, test_seq_len, test_lora_rank), dtype=dtype, device='cuda')
    print(f"Compress KV shape: {test_compressed_kv.shape}")
    
    # Burn-in runs
    for _ in range(burn_in):
        _ = test_simple_compressed_attn(test_hidden_q, test_compressed_kv)
        torch.cuda.synchronize()
    
    # Time over 10 runs
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        output = test_simple_compressed_attn(test_hidden_q, test_compressed_kv)
        torch.cuda.synchronize()
        total_time += time.time() - start_time
    
    avg_time = total_time / num_runs
    print(f"Output shape: {output.shape}")
    print(f"Time taken (avg of {num_runs} runs, after {burn_in} burn-in): {avg_time:.6f} seconds")
    

    ############################################################
    # SimpleAbsorbedAttention   
    ############################################################
    # Burn-in runs
    for _ in range(burn_in):
        _ = test_simple_absorbed_attn(test_hidden_q, test_compressed_kv)
        torch.cuda.synchronize()
    
    # Time over 10 runs
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        output = test_simple_absorbed_attn(test_hidden_q, test_compressed_kv)
        torch.cuda.synchronize()
        total_time += time.time() - start_time
    
    avg_time = total_time / num_runs
    print(f"Output shape: {output.shape}")
    print(f"SimpleAbsorbedAttention time (avg of {num_runs} runs, after {burn_in} burn-in): {avg_time:.6f} seconds")

