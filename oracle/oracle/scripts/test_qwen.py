import os
import torch
import torch.nn as nn

from utils import DeepSeekDistillation
from models import MyQwen3ForCausalLM # 这里还需要 models/myqwen.py, qwen3config.py 文件
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup, AutoModelForCausalLM
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from transformer_engine.pytorch import fp8_autocast


BATCH_SIZE = 4
SEQ_LEN = 1024

DATA_SHUFFLE = True
FREEZE_TEACHER = True

USE_BF16 = True

# MoE 的参数，目前训练的 2.8B-0.8B 模型的参数就是如下
USE_MOE = True
MOE_INTERMEDIATE_SIZE = 1536
EXPERT_PER_TOKEN = 2
NUM_EXPERTS = 16
GATING_REFERENCE = "oracle" # oracle / switch


DEVICE = 0
CONFIG_DIR = "../../Qwen-family/Qwen3-0.6B" # 就直接用 qwen3 的 dir 就可以
CKPT_DIR = "../checkpoints/qwen/2.8-0.8-from-official-Qwen3-0.6B" # checkpoint dir
TEST_CHECKPOINT = "0.20000.pth" # test checkpoint
DATA_DIR = "../../washed_data_0520/train" # 这里是我用的 test data，下游任务测试应该不需要。


# 这个是计算专家激活差别的，这个我来算就好了
def _cal_avg_expert_change(expert_labels):
    change = []
    pre = set(expert_labels[0].cpu().numpy())
    for i in range(1, len(expert_labels)):
        cur = set(expert_labels[i].cpu().numpy())
        intersect = pre.intersection(cur)
        change.append(len(intersect) / len(pre))
        pre = cur
    return sum(change) / len(change)


@torch.no_grad()
def prepare_model():
    print("Construct student model. 1")

    config = AutoConfig.from_pretrained(CONFIG_DIR, trust_remote_code = True)
    config.moe_intermediate_size = MOE_INTERMEDIATE_SIZE
    config.num_experts_per_tok = EXPERT_PER_TOKEN
    config.num_experts = NUM_EXPERTS
    config.gating_reference = GATING_REFERENCE
    config.norm_topk_prob = True
    config.use_moe = USE_MOE

    print("Construct student model.2")

    model = MyQwen3ForCausalLM(config).to(DEVICE)
    print("Construct student model. 3")
    # print("Construct student model. 4")

    print(f'rank {DEVICE} student model ok, params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B/{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B') # 
    return model


def prepare_data():
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)
    dataset = DeepSeekDistillation(DATA_DIR, SEQ_LEN, tokenizer)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=DATA_SHUFFLE)

    return dataloader


def prepare_loss_optimizer(s_model):
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=151643, reduction='mean')
    
    return token_loss_fn


@torch.no_grad()
def forward_step(local_batch_idx, source, target, model, token_loss_fn):
    source, target = source.to(DEVICE), target.to(DEVICE)
    batch_size, seq_length = source.shape[:2]
    past_key_values_length = 0
    position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=DEVICE)
    position_ids = position_ids.unsqueeze(0).to(DEVICE)

    s_output = model(source, position_ids=position_ids, output_hidden_states = True, output_expert_labels = True)
    s_logits = s_output.logits
    
    for layer_idx, expert_label in enumerate(s_output.expert_labels):
        print(f"layer {layer_idx} expert intersect: {_cal_avg_expert_change(expert_label)}")
    loss = token_loss_fn(s_logits.view(-1, s_logits.size(-1)), target.reshape(-1))

    print(f"batch: {local_batch_idx}, loss: {loss:.3f}", flush=True)

    return loss


def update_step(optimizer, scheduler):
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

import matplotlib.pyplot as plt


def draw_singular_values(values, title, fig_path):
    for k, v in values.items():
        plt.plot(v, label=f"step {k}")
    plt.title(title)
    plt.legend()
    plt.savefig(fig_path)
    plt.close()


@torch.no_grad()
def analyse_expert_singularity(model, batch, sig_val_dict):
    for layer_idx, layer in enumerate(model.model.layers):
        expert = layer.mlp.shared_expert
        u, s, v = torch.linalg.svd(expert.up_proj.weight)
        sig_val_dict[f"layer.{layer_idx}.shared_expert.up"][batch] = s.cpu().numpy()
        
        u, s, v = torch.linalg.svd(expert.down_proj.weight)
        sig_val_dict[f"layer.{layer_idx}.shared_expert.down"][batch] = s.cpu().numpy()
        
        u, s, v = torch.linalg.svd(expert.gate_proj.weight)
        sig_val_dict[f"layer.{layer_idx}.shared_expert.gate"][batch] = s.cpu().numpy()

        for expert_idx, expert in enumerate(layer.mlp.experts):
            u, s, v = torch.linalg.svd(expert.up_proj.weight)
            sig_val_dict[f"layer.{layer_idx}.expert{expert_idx}.up"][batch] = s.cpu().numpy()
            
            u, s, v = torch.linalg.svd(expert.down_proj.weight)
            sig_val_dict[f"layer.{layer_idx}.expert{expert_idx}.down"][batch] = s.cpu().numpy()
            
            u, s, v = torch.linalg.svd(expert.gate_proj.weight)
            sig_val_dict[f"layer.{layer_idx}.expert{expert_idx}.gate"][batch] = s.cpu().numpy()
        #     if expert_idx > 2:
        #         break
        
        # if layer_idx > 2:
        #     break

@torch.no_grad()
def thread_main():
    print(f"running on DEVICE {DEVICE}")
    model = prepare_model() # 测试下游任务就只要用这一个函数就好了。
    
    sig_val_dict = {}
    for l in range(28):
        for mat in ["up","down","gate"]:
            sig_val_dict[f"layer.{l}.shared_expert.{mat}"] = {}
            for e in range(16):
                sig_val_dict[f"layer.{l}.expert{e}.{mat}"] = {}
            
    for test_epoch in range (0, 1):
        for test_ckpt in range(0, 24000 + 1, 3000):
            print(f"test batch {test_ckpt}")
            if test_ckpt > 0:
                if os.path.exists(f"{CKPT_DIR}/0.{test_ckpt}.pth"):
                    model.load_state_dict(torch.load(f"{CKPT_DIR}/0.{test_ckpt}.pth", weights_only = True, map_location="cpu"))
                else:
                    break
            analyse_expert_singularity(model, test_ckpt, sig_val_dict)
            # if test_ckpt > 3000:
            #     break

    for layer_idx, layer in enumerate(model.model.layers):
        for mat in ["up", "down", "gate"]:
            draw_singular_values(sig_val_dict[f"layer.{layer_idx}.shared_expert.{mat}"],
                                    f"layer.{layer_idx}.shared_expert.{mat}",
                                    f"../figures/singular_values/{mat}.layer.{layer_idx}.shared_expert.png")
            for expert_idx, expert in enumerate(layer.mlp.experts):
                draw_singular_values(sig_val_dict[f"layer.{layer_idx}.expert{expert_idx}.{mat}"],
                                     f"layer.{layer_idx}.expert{expert_idx}.{mat}",
                                     f"../figures/singular_values/{mat}.layer.{layer_idx}.expert{expert_idx}.png")

    # dataloader = prepare_data()
    # token_loss_fn = prepare_loss_optimizer(model)
        
    # for local_batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
    #     loss = forward_step(local_batch_idx, source, target, model, token_loss_fn)


if __name__ == "__main__":
    thread_main()
    
