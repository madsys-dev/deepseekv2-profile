This patched DeepseekV2Model contains the following modifications to DeepseekV2Attention:

1. Instead of caching the decompressed key/value states, we cache the low-rank key-value joint compression as well as 
the decoupled RoPE part of the keys. For the sake of reusing the cache utility of transformers library only, we treat 
k_pe as key_states and compressed_kv as value_states.
2. We add the absorption technique described in the DeepseekV2 paper, by changing the multiplication order when 
computing query and output vectors. This not only saves memory consumption of intermediate tensors but also reduces 
the number of floating-point operations.
3. We compute the RoPE part and non-RoPE part of the attention score separately and then sum them up. The original 
implementation concatenates the two parts of the query/key vectors, which has proven to be quite inefficient when 
caching compressed key/value states due to unnecessary data broadcast and memory round-trips.
