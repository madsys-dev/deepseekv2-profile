import torch
import time

class SimpleAttention(torch.nn.Module):
    def __init__(self, num_heads: int, head_dim: int, q_lora_rank: int, dtype=torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim
        self.q_lora_rank = q_lora_rank
        self.dtype = dtype
        
        self.q_proj = torch.nn.Linear(q_lora_rank, num_heads * head_dim, dtype=dtype, device='cuda')
        
    def forward(self, hidden_states_q: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor):
        """
        Dimension key:

        B: batch size
        L: sequence length (number of past tokens)
        D: model dimension (also called d_model or embedding_dim)
        H: number of attention heads in a layer
        K: size of each attention key or value per head (d_kv, typically D / H)
        C: latent dimension (d_c)

        Input:
        - hidden_state: [B, D]
        - key_states: [B, L, H, K]
        - value_states: [B, L, H, K]
        """
        bsz = hidden_states_q.shape[0]
        
        query_B_H_K = self.q_proj(hidden_states_q).view(bsz, self.num_heads, self.head_dim)

        key_B_H_L_K = key_states.permute(0, 2, 1, 3)
        value_B_H_L_K = value_states.permute(0, 2, 1, 3)
        query_B_H_1_K = query_B_H_K.unsqueeze(2) 
        
        attn_scores_B_H_1_L = torch.matmul(query_B_H_1_K, key_B_H_L_K.transpose(-1, -2))  
        
        attn_probs_B_H_1_L = torch.nn.functional.softmax(attn_scores_B_H_1_L, dim=-1)
        attn_output_B_H_1_K = torch.matmul(attn_probs_B_H_1_L, value_B_H_L_K)

        attn_output_B_D = attn_output_B_H_1_K.squeeze(2).reshape(bsz, self.hidden_size)
        
        return attn_output_B_D


class SimpleCompressedAttention(torch.nn.Module):
    def __init__(self, num_heads: int, head_dim: int, lora_rank: int, q_lora_rank: int, dtype=torch.float32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lora_rank = lora_rank
        self.q_lora_rank = q_lora_rank
        self.dtype = dtype
        
        self.q_proj = torch.nn.Linear(q_lora_rank, num_heads * head_dim, dtype=dtype, device='cuda')
        self.k_proj = torch.nn.Linear(lora_rank, num_heads * head_dim, dtype=dtype, bias=False, device='cuda')
        self.v_proj = torch.nn.Linear(lora_rank, num_heads * head_dim, dtype=dtype, bias=False, device='cuda')
        
    def forward(self, hidden_states_q, compressed_kv):
        """
        Input:
        - hidden_states_q: [B, D]
        - compressed_kv: [B, L, C]
        """
        bsz = hidden_states_q.shape[0]
        seq_len = compressed_kv.shape[1]
        
        query_B_H_K = self.q_proj(hidden_states_q).view(bsz, self.num_heads, self.head_dim)
        
        key_B_L_HD = self.k_proj(compressed_kv) 
        key_B_L_H_K = key_B_L_HD.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_B_H_L_K = key_B_L_H_K.permute(0, 2, 1, 3).contiguous()  
        
        value_B_L_HD = self.v_proj(compressed_kv) 
        value_B_L_H_K = value_B_L_HD.view(bsz, seq_len, self.num_heads, self.head_dim)
        value_B_H_L_K = value_B_L_H_K.permute(0, 2, 1, 3).contiguous()  
        
        query_BH_1_K = query_B_H_K.reshape(bsz * self.num_heads, 1, self.head_dim)
        key_BH_L_K = key_B_H_L_K.reshape(bsz * self.num_heads, seq_len, self.head_dim)
        key_BH_K_L = key_BH_L_K.transpose(1, 2)
        attn_weights_BH_1_L = torch.bmm(query_BH_1_K, key_BH_K_L)
        attn_weights_B_H_L = attn_weights_BH_1_L.view(bsz, self.num_heads, seq_len)
        
        attn_probs_B_H_L = torch.nn.functional.softmax(attn_weights_B_H_L, dim=-1)
        
        attn_probs_BH_1_L = attn_probs_B_H_L.reshape(bsz * self.num_heads, 1, seq_len)
        value_BH_L_K = value_B_H_L_K.reshape(bsz * self.num_heads, seq_len, self.head_dim)
        attn_output_BH_1_K = torch.bmm(attn_probs_BH_1_L, value_BH_L_K)
        attn_output_B_H_K = attn_output_BH_1_K.view(bsz, self.num_heads, self.head_dim)
        
        # Reshape to final output
        attn_output_B_D = attn_output_B_H_K.reshape(bsz, self.num_heads * self.head_dim)
        
        return attn_output_B_D

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
        - hidden_states_q: [B, D]
        - compressed_kv: [B, L, C]
        """
        bsz = hidden_states_q.shape[0]
        
        query_B_H_K = self.q_proj(hidden_states_q).view(bsz, self.num_heads, self.head_dim)
        
        query_reshaped_B_H_K_1 = query_B_H_K.view(bsz, self.num_heads, self.head_dim, 1)
        query_absorbed_B_H_R = torch.matmul(self.w_kc.transpose(1, 2), query_reshaped_B_H_K_1).squeeze(-1)
        
        compressed_kv_transposed_B_R_L = compressed_kv.transpose(1, 2).contiguous()
        attn_scores_B_H_L = torch.einsum('bhr,brl->bhl', query_absorbed_B_H_R, compressed_kv_transposed_B_R_L)
        attn_probs_B_H_L = torch.nn.functional.softmax(attn_scores_B_H_L, dim=-1)
        
        context_B_H_R = torch.einsum('bhl,blr->bhr', attn_probs_B_H_L, compressed_kv)
        context_reshaped_B_H_R_1 = context_B_H_R.view(bsz, self.num_heads, self.lora_rank, 1)
        attn_output_B_H_K = torch.matmul(self.w_vc, context_reshaped_B_H_R_1).squeeze(-1)
        
        attn_output_B_D = attn_output_B_H_K.reshape(bsz, self.num_heads * self.head_dim)
        
        return attn_output_B_D


if __name__ == "__main__":
    print("\nRunning attention comparison test...")

    # Set CUDA device to GPU 5
    torch.cuda.set_device(5)
    print(f"Using CUDA device: {torch.cuda.current_device()}")

    torch.set_grad_enabled(False)
    test_batch_size = 1
    test_seq_len = 10000
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
    torch.manual_seed(42)  

    # Initialize SimpleCompressedAttention
    compressed_attn = SimpleCompressedAttention(
        num_heads=test_heads, 
        head_dim=head_dim, 
        lora_rank=test_lora_rank, 
        q_lora_rank=q_lora_rank, 
        dtype=dtype
    ).cuda()
    
    with torch.no_grad():
        compressed_attn.q_proj.weight.uniform_(4, 5)
        compressed_attn.k_proj.weight.uniform_(4, 5)
        compressed_attn.v_proj.weight.uniform_(4, 5)
    
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
    
    tolerance = 1e-5
    is_close = max_diff < tolerance
    print(f"Outputs are {'equivalent' if is_close else 'different'} (tolerance: {tolerance})")
    
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
