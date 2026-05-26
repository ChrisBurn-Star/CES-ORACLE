import os
import json
import torch
import torch.nn as nn

from models import MyDeepseekV2MoE
from utils import DeepSeekDistillation

from accelerate import infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

BATCH_SIZE = 8
SEQ_LEN = 2048
EXPERT_NUM = 64
TOPK_EXPERT = 6
T_DTYPE = torch.bfloat16
S_DTYPE = torch.float32
GATING_REFERENCE = 'attn_output'
LLM_DIR = "/inspire/hdd/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/yunweiyuhuifu/p-shangli/DeepseekV2-16B-A2.8B"

class MyIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
        **kwargs):

        outputs = (hidden_states,)
        expert_label = None

        if output_attentions:
            outputs += (None,)

        if use_cache:
            outputs += (None,)
        
        if expert_label is not None:
            outputs += (expert_label,)

        return outputs

@torch.no_grad()
def prepare_teacher(replace_layers, weight_dirs = None):
    with open(f"{LLM_DIR}/device_map.json", 'r') as f:
        device_map = dict(json.load(f))
    t_model = AutoModelForCausalLM.from_pretrained(LLM_DIR, torch_dtype = T_DTYPE, local_files_only = True, trust_remote_code=True, device_map = device_map)
    t_model.eval()
    t_model.requires_grad_(False)
    print(f'teacher model loaded.')


    for replace_layer, weight_dir in zip(replace_layers, weight_dirs):
        if weight_dir is None:
            # 重新随机初始化 teacher 的 MoE
            for params in t_model.model.layers[replace_layer].mlp.parameters():
                nn.init.uniform_(params, 0, .01)
        else:
            # 替换为我蒸馏后的 MoE 参数
            s_model = MyDeepseekV2MoE(2048, 1408 * 2, 1408, 'silu', EXPERT_NUM, TOPK_EXPERT, .001)
            s_model.load_state_dict(torch.load(weight_dir, map_location='cpu', weights_only=True), strict=False)
            s_model = s_model.to(T_DTYPE).to(t_model.model.layers[replace_layer].input_layernorm.weight.device)
            t_model.model.layers[replace_layer].mlp = s_model
    
    return t_model



if __name__ == '__main__':
    # replace_layers 里的 layer index 从 1 开始，其中模型的第 1 层是一个 dense，第 2 - 27 层是 MoE。
    # replace_layers 是一个列表，因为之前训练的时候可能希望把第 2、3 层蒸馏成一个新层，因此 [2, 3] 就会把原有第 2 层变成我们蒸馏的结果，然后第 3 层直接删掉，这样相当于把两层合并成一层。
    # replace_layers 也可以只输入一个元素，比如 [2]，这时候就只把第 2 层换掉，其余不动。
    # weight_dir 是要加载的我们的参数，checkpoints 目录下 layer_xx_yy 就是把第 xx-yy 层一起蒸馏成一个新层，后缀有 oracle_gate 就是用我们的方法初始化，否则就是随机初始化。
    # checkpoints 下面的所有模型都是我们的 gating 机制，只是初始化有分别。
    # weight_dir 参数也可以为 None，为 None 时就直接把 replace_layers 里指定的层重新高斯初始化
    # 上面这些功能就可以让我们测试包括随机重置某层、换成我们的模型之类的下游任务效果了。

    # 把原模型的第 2、4、8 层换成我们训练的层：
    t_model = prepare_teacher(replace_layers = [2, 4, 8], weight_dirs=[
        "../checkpoints/layer_2_2_oracle_gate/12000.pth",
        "../checkpoints/layer_4_4_oracle_gate/15000.pth",
        "../checkpoints/layer_8_8_oracle_gate/10000.pth"
        ])

    tokenizer = AutoTokenizer.from_pretrained(LLM_DIR, trust_remote_code=True)
    dataset = DeepSeekDistillation('../../washed_data_0511/train', SEQ_LEN, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for batch_idx, (source, target, realens) in enumerate(dataloader, 1):
        batch_size, seq_length = source.shape[:2]
        past_key_values_length = 0
        position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=source.device,)
        position_ids = position_ids.unsqueeze(0)

        # t = time.time()
        with torch.no_grad():
            t_output = t_model(source, position_ids=position_ids, output_hidden_states = True)

        print(batch_idx)