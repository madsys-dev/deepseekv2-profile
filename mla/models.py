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
        self.q = q
        
        # Weight absorption trick: FIXED to match the working version
        # Use matmul with reshaping instead of bmm+transpose
        q_reshaped = q.view(bsz, self.num_heads, self.head_dim, 1)
        q_absorbed = torch.matmul(self.w_kc.transpose(1, 2), q_reshaped).squeeze(-1)
        
        # Calculate attention scores without expansion
        # Transpose compressed_kv to [bsz, lora_rank, seq_len]
        compressed_kv_transposed = compressed_kv.transpose(1, 2).contiguous()
        
        # For each batch and head, compute attention scores
        # Reshape q_absorbed to [bsz, num_heads, 1, lora_rank]
        # Use regular matmul with broadcasting
        attn_scores = torch.matmul(
            q_absorbed.unsqueeze(2),  # [bsz, num_heads, 1, lora_rank]
            compressed_kv_transposed.unsqueeze(1)  # [bsz, 1, lora_rank, seq_len]
        ).squeeze(2)  # Result: [bsz, num_heads, seq_len]
        
        # Apply softmax to get attention probabilities
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to compressed values without expansion
        # Reshape attn_probs to [bsz, num_heads, seq_len, 1]
        # Use regular matmul with broadcasting
        context = torch.matmul(
            attn_probs.unsqueeze(3),  # [bsz, num_heads, seq_len, 1]
            compressed_kv.unsqueeze(1)  # [bsz, 1, seq_len, lora_rank]
        ).squeeze(2)  # Result: [bsz, num_heads, lora_rank]
        
        # Final projection: FIXED to match the working version
        # Use matmul with reshaping instead of bmm+transpose
        context_reshaped = context.view(bsz, self.num_heads, self.lora_rank, 1)
        attn_output = torch.matmul(self.w_vc, context_reshaped).squeeze(-1)
        
        # Reshape to final output
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
        
        # Separate projections for key and value
        self.k_proj = torch.nn.Linear(lora_rank, num_heads * head_dim, dtype=dtype, bias=False, device='cuda')
        self.v_proj = torch.nn.Linear(lora_rank, num_heads * head_dim, dtype=dtype, bias=False, device='cuda')
        self.q = None
    def forward(self, hidden_states_q, compressed_kv):
        """
        Input:
        - hidden_states_q: [bsz, q_lora_rank]
        - compressed_kv: [bsz, seq_len, lora_rank]
        """
        bsz = hidden_states_q.shape[0]
        seq_len = compressed_kv.shape[1]
        
        # Project hidden_states_q
        q = self.q_proj(hidden_states_q)
        q = q.view(bsz, self.num_heads, self.head_dim)
        self.q = q
        
        # Project compressed_kv to key and value separately
        k = self.k_proj(compressed_kv)  # [bsz, seq_len, num_heads * head_dim]
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3).contiguous()  # [bsz, num_heads, seq_len, head_dim]
        
        v = self.v_proj(compressed_kv)  # [bsz, seq_len, num_heads * head_dim]
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3).contiguous()  # [bsz, num_heads, seq_len, head_dim]
        
        # Compute attention scores using bmm
        q_reshaped = q.reshape(bsz * self.num_heads, 1, self.head_dim)
        k_reshaped = k.reshape(bsz * self.num_heads, seq_len, self.head_dim).transpose(1, 2)
        attn_weights = torch.bmm(q_reshaped, k_reshaped).view(bsz, self.num_heads, seq_len)
        
        # Apply softmax to get attention probabilities
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # Apply attention to value using bmm
        attn_probs_reshaped = attn_probs.reshape(bsz * self.num_heads, 1, seq_len)
        v_reshaped = v.reshape(bsz * self.num_heads, seq_len, self.head_dim)
        attn_output = torch.bmm(attn_probs_reshaped, v_reshaped).view(bsz, self.num_heads, self.head_dim)
        
        # Reshape to final output
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
        
        # Query projection
        self.q_proj = torch.nn.Linear(q_lora_rank, num_heads * head_dim, dtype=dtype, device='cuda')
        self.k_proj = torch.nn.Linear(lora_rank, num_heads * head_dim, dtype=dtype, bias=False, device='cuda')
        self.v_proj = torch.nn.Linear(lora_rank, num_heads * head_dim, dtype=dtype, bias=False, device='cuda')
        
        # Precompute the weight matrices (keep this part)
        self.w_kc = self.k_proj.weight.view(num_heads, head_dim, lora_rank).contiguous()
        self.w_vc = self.v_proj.weight.view(num_heads, head_dim, lora_rank).contiguous()
        
        self.q = None
    
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
        self.q = q
        
        # Weight absorption trick: FIXED to match the working version
        # Use matmul with reshaping instead of bmm+transpose
        q_reshaped = q.view(bsz, self.num_heads, self.head_dim, 1)
        q_absorbed = torch.matmul(self.w_kc.transpose(1, 2), q_reshaped).squeeze(-1)
        
        # Calculate attention scores: OPTIMIZED to avoid expansion
        # Transpose first to get [bsz, lora_rank, seq_len]
        compressed_kv_transposed = compressed_kv.transpose(1, 2).contiguous()
        
        # Reshape q_absorbed to [bsz, num_heads, lora_rank]
        # Reshape compressed_kv_transposed to [bsz, 1, lora_rank, seq_len]
        # Use einsum for efficient batched matrix multiplication without expansion
        attn_scores = torch.einsum('bhl,blk->bhk', q_absorbed, compressed_kv_transposed)
        
        # Apply softmax to get attention probabilities
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to compressed values: OPTIMIZED to avoid expansion
        # Use einsum instead of bmm with expansion
        context = torch.einsum('bhs,bsl->bhl', attn_probs, compressed_kv)
        
        # Final projection: FIXED to match the working version
        # Use matmul with reshaping instead of bmm+transpose
        context_reshaped = context.view(bsz, self.num_heads, self.lora_rank, 1)
        attn_output = torch.matmul(self.w_vc, context_reshaped).squeeze(-1)
        
        # Reshape to final output
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim)
        
        return attn_output

# Add this at the bottom of the file
if __name__ == "__main__":
    print("\nRunning attention comparison test...")

    # Set CUDA device to GPU 5
    torch.cuda.set_device(5)
    print(f"Using CUDA device: {torch.cuda.current_device()}")

    torch.set_grad_enabled(False)
    # Create a simple test case
    test_batch_size = 1
    test_seq_len = 1000
    test_heads = 320
    head_dim = 8
    test_lora_rank = 8
    q_lora_rank = 1536

    # Choose precision
    dtype = torch.float32  # Using float32 for more precise comparison
    
    # Create test inputs with specified precision
    test_hidden_q = torch.randn((test_batch_size, q_lora_rank), dtype=dtype, device='cuda')
    test_compressed_kv = torch.randn((test_batch_size, test_seq_len, test_lora_rank), dtype=dtype, device='cuda')
    
    # Create models with identical initialization
    torch.manual_seed(42)  # Set seed for reproducibility
    
    # Initialize SimpleCompressedAttention
    compressed_attn = SimpleCompressedAttention(
        num_heads=test_heads, 
        head_dim=head_dim, 
        lora_rank=test_lora_rank, 
        q_lora_rank=q_lora_rank, 
        dtype=dtype
    ).cuda()
    
    # Save weights to initialize SimpleAbsorbedAttention with the same values
    q_proj_weight = compressed_attn.q_proj.weight.clone()
    k_proj_weight = compressed_attn.k_proj.weight.clone()
    v_proj_weight = compressed_attn.v_proj.weight.clone()
    
    # Initialize SimpleAbsorbedAttention with the same weights
    absorbed_attn = SimpleAbsorbedAttention(
        num_heads=test_heads, 
        head_dim=head_dim, 
        lora_rank=test_lora_rank, 
        q_lora_rank=q_lora_rank, 
        dtype=dtype
    ).cuda()
    
    # Copy q_proj weights directly
    absorbed_attn.q_proj.weight.data.copy_(q_proj_weight)
    absorbed_attn.k_proj.weight.data.copy_(k_proj_weight)
    absorbed_attn.v_proj.weight.data.copy_(v_proj_weight)
    # For the kv weights, we need to reshape and copy to match the absorbed attention format
        # Verify weights are identical
        # Update the reshaped weight matrices after copying the weights
    absorbed_attn.w_kc = absorbed_attn.k_proj.weight.view(test_heads, head_dim, test_lora_rank).contiguous()
    absorbed_attn.w_vc = absorbed_attn.v_proj.weight.view(test_heads, head_dim, test_lora_rank).contiguous()

    # Run both models
    with torch.no_grad():
        compressed_output = compressed_attn(test_hidden_q, test_compressed_kv)
        absorbed_output = absorbed_attn(test_hidden_q, test_compressed_kv)

    # Compare outputs
    max_diff = torch.max(torch.abs(compressed_output - absorbed_output)).item()
    avg_diff = torch.mean(torch.abs(compressed_output - absorbed_output)).item()
    
    print(f"Output shapes - Compressed: {compressed_output.shape}, Absorbed: {absorbed_output.shape}")
    print(f"Maximum absolute difference: {max_diff:.6e}")
    print(f"Average absolute difference: {avg_diff:.6e}")
    
    # Check if outputs are close enough (allowing for numerical precision differences)
    tolerance = 1e-5
    is_close = max_diff < tolerance
    print(f"Outputs are {'equivalent' if is_close else 'different'} (tolerance: {tolerance})")
    
    # Compare performance
    num_runs = 10
    burn_in = 2
    
    # Burn-in runs for compressed attention
    for _ in range(burn_in):
        _ = compressed_attn(test_hidden_q, test_compressed_kv)
        torch.cuda.synchronize()
    
    # Time compressed attention
    total_time_compressed = 0
    for _ in range(num_runs):
        start_time = time.time()
        _ = compressed_attn(test_hidden_q, test_compressed_kv)
        torch.cuda.synchronize()
        total_time_compressed += time.time() - start_time
    
    avg_time_compressed = total_time_compressed / num_runs
    
    # Burn-in runs for absorbed attention
    for _ in range(burn_in):
        _ = absorbed_attn(test_hidden_q, test_compressed_kv)
        torch.cuda.synchronize()
    
    # Time absorbed attention
    total_time_absorbed = 0
    for _ in range(num_runs):
        start_time = time.time()
        _ = absorbed_attn(test_hidden_q, test_compressed_kv)
        torch.cuda.synchronize()
        total_time_absorbed += time.time() - start_time
    
    avg_time_absorbed = total_time_absorbed / num_runs
    
    print(f"SimpleCompressedAttention time: {avg_time_compressed:.6f} seconds")
    print(f"SimpleAbsorbedAttention time: {avg_time_absorbed:.6f} seconds")
    print(f"Speedup: {avg_time_compressed / avg_time_absorbed:.2f}x")

# # Add this at the bottom of the file
# if __name__ == "__main__":
#     print("\nRunning SimpleAttention test...")

#     # Set CUDA device to GPU 5
#     torch.cuda.set_device(5)
#     print(f"Using CUDA device: {torch.cuda.current_device()}")

#     torch.set_grad_enabled(False)
#     # Create a simple test case
#     test_batch_size = 2
#     test_seq_len = 10
#     test_heads = 128#32
#     head_dim = 5120
#     test_lora_rank = 512
#     q_lora_rank = 1536

#     # Choose precision
#     dtype = torch.float16  # or torch.bfloat16 or torch.float32
    
#     # Create test inputs with specified precision
#     test_hidden_q = torch.randn((test_batch_size, q_lora_rank), dtype=dtype, device='cuda')
#     #test_key = torch.randn((test_batch_size, test_seq_len, test_heads, head_dim), dtype=dtype, device='cuda')
#     #test_value = torch.randn((test_batch_size, test_seq_len, test_heads, head_dim), dtype=dtype, device='cuda')
    
#     # Create model with specified precision
#     #test_simple_attn = SimpleAttention(num_heads=test_heads, head_dim=head_dim, dtype=dtype, q_lora_rank=q_lora_rank).cuda()
#     #test_simple_compressed_attn = SimpleCompressedAttention(num_heads=test_heads, head_dim=head_dim, lora_rank=test_lora_rank, q_lora_rank=q_lora_rank, dtype=dtype).cuda()
#     test_simple_absorbed_attn = SimpleAbsorbedAttention(num_heads=test_heads, head_dim=head_dim, lora_rank=test_lora_rank, q_lora_rank=q_lora_rank, dtype=dtype).cuda()

#     # Run forward pass with timing (average of 10 runs)
#     # print(f"Input shape: {test_hidden_q.shape}")
#     # print(f"Key shape: {test_key.shape}")
#     # print(f"Value shape: {test_value.shape}")
    
#     # Burn-in runs
#     burn_in = 2
#     # for _ in range(burn_in):
#     #     _ = test_simple_attn(test_hidden_q, test_key, test_value)
#     #     torch.cuda.synchronize()
    
#     # Time over 10 runs
#     num_runs = 10
#     total_time = 0
#     # for _ in range(num_runs):
#     #     start_time = time.time()
#     #     output = test_simple_attn(test_hidden_q, test_key, test_value)
#     #     torch.cuda.synchronize()
#     #     total_time += time.time() - start_time
    
#     # avg_time = total_time / num_runs
#     # print(f"Output shape: {output.shape}")
#     # print(f"Time taken (avg of {num_runs} runs, after {burn_in} burn-in): {avg_time:.6f} seconds")

#     # # Run compressed attention with timing (average of 10 runs)
#     test_compressed_kv = torch.randn((test_batch_size, test_seq_len, test_lora_rank), dtype=dtype, device='cuda')
#     # print(f"Compress KV shape: {test_compressed_kv.shape}")
    
#     # Burn-in runs
#     # for _ in range(burn_in):
#     #     _ = test_simple_compressed_attn(test_hidden_q, test_compressed_kv)
#     #     torch.cuda.synchronize()
    
#     # # Time over 10 runs
#     # total_time = 0
#     # for _ in range(num_runs):
#     #     start_time = time.time()
#     #     output = test_simple_compressed_attn(test_hidden_q, test_compressed_kv)
#     #     torch.cuda.synchronize()
#     #     total_time += time.time() - start_time
    
#     # avg_time = total_time / num_runs
#     # print(f"Output shape: {output.shape}")
#     # print(f"Time taken (avg of {num_runs} runs, after {burn_in} burn-in): {avg_time:.6f} seconds")
    

#     ############################################################
#     # SimpleAbsorbedAttention   
#     ############################################################
#     # Burn-in runs

#     print(f"Compress KV shape: {test_compressed_kv.shape}")
#     for _ in range(burn_in):
#         _ = test_simple_absorbed_attn(test_hidden_q, test_compressed_kv)
#         torch.cuda.synchronize()
    
#     # Time over 10 runs
#     total_time = 0
#     for _ in range(num_runs):
#         start_time = time.time()
#         output = test_simple_absorbed_attn(test_hidden_q, test_compressed_kv)
#         torch.cuda.synchronize()
#         total_time += time.time() - start_time
    
#     avg_time = total_time / num_runs
#     print(f"Output shape: {output.shape}")
#     print(f"SimpleAbsorbedAttention time (avg of {num_runs} runs, after {burn_in} burn-in): {avg_time:.6f} seconds")

