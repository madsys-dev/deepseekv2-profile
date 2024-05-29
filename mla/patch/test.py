import torch

from transformers import GenerationConfig

from .configuration_deepseek import DeepseekV2Config
from ..modeling_deepseek import DeepseekV2ForCausalLM
from .modeling_deepseek import DeepseekV2ForCausalLM as PatchedDeepseekV2ForCausalLM

torch.set_grad_enabled(False)
# torch.set_default_dtype(torch.bfloat16)

config = DeepseekV2Config.from_json_file('mla/patch/reduced_layer_config.json')
model = DeepseekV2ForCausalLM(config=config).cuda(0)

generation_config = GenerationConfig.from_model_config(config)
generation_config.pad_token_id = generation_config.eos_token_id
generation_config.do_sample = False
model.generation_config = generation_config

patched_model = PatchedDeepseekV2ForCausalLM(config=config).cuda(1)
patched_model.load_state_dict(model.state_dict())
patched_model.generation_config = model.generation_config

in_t = torch.randint(100, 200, (32, 64), dtype=torch.long).cuda(0)

out_t = model.generate(in_t, max_new_tokens=10)
patched_out_t = patched_model.generate(in_t.cuda(1), max_new_tokens=10)

diff = torch.ne(out_t, patched_out_t.cuda(0))
diff_elts = diff.count_nonzero()
print(f'{diff_elts} elements differ: {diff.nonzero()}')
