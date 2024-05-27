from typing import Optional
import torch
from torch import nn

class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = max_position_embeddings

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q: torch.Tensor, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        # k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

class DeepseekAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, q_lora_rank: int, qk_rope_head_dim: int, 
                 kv_lora_rank: int, v_head_dim: int, qk_nope_head_dim: int, max_position_embeddings: int,
                  torch_dtype: torch.dtype, attention_bias: bool = False, *args, **kwargs):
        super().__init__()
        q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.q_head_dim = q_head_dim
        self.v_head_dim = v_head_dim
        self.softmax_scale = torch.tensor(self.q_head_dim).to(torch_dtype).rsqrt()
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=attention_bias, dtype=torch_dtype)
        self.q_a_layernorm = DeepseekV2RMSNorm(q_lora_rank).to(torch_dtype)
        self.q_b_proj = nn.Linear(q_lora_rank, num_attention_heads * q_head_dim, bias=False, dtype=torch_dtype)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=attention_bias, dtype=torch_dtype)
        self.kv_a_layernorm = DeepseekV2RMSNorm(kv_lora_rank).to(torch_dtype)
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_attention_heads * (qk_nope_head_dim + v_head_dim), bias=False, dtype=torch_dtype)
        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=attention_bias, dtype=torch_dtype)
        self.rotary_emb = DeepseekV2RotaryEmbedding(self.qk_rope_head_dim, max_position_embeddings=max_position_embeddings).to(torch_dtype)

    def forward(self, hidden_states_q: torch.Tensor, hidden_states_kv: torch.Tensor, q_position_ids: torch.LongTensor, 
                kv_position_ids: torch.LongTensor):
        '''
        Attention masks and past cache are removed.

        Input: 
        - hidden_states_q: [bsz, q_len, hidden_size]
        - hidden_states_kv: [bsz, kv_len, hidden_size]
        - position_ids: [bsz, q_len]
        '''
        bsz, q_len, _ = hidden_states_q.size()
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states_q)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        kv_seq_len = hidden_states_kv.size(1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states_kv)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, kv_seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        
        kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
        q_absorb = kv_b_proj[:, :self.qk_nope_head_dim,:]
        out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]
        
        cos, sin = self.rotary_emb(q_pe)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, q_position_ids)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, kv_position_ids)

        qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, qk_head_dim)
        

        query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
        query_states[:, :, :, self.kv_lora_rank :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, kv_seq_len, qk_head_dim)
        key_states[:, :, :, : self.kv_lora_rank] = compressed_kv.unsqueeze(1)
        key_states[:, :, :, self.kv_lora_rank :] = k_pe

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.einsum('bhqs,bsD->bhqD', attn_weights, compressed_kv)
        attn_output = torch.einsum('bhqD,hdD->bhqd', attn_output, out_absorb)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output
