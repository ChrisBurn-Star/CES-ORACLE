import os
import time
import numpy as np
import torch
import torch.nn as nn

import random
from utils import DeepSeekDistillation
from models import MyQwen3ForCausalLM # 这里还需要 models/myqwen.py, qwen3config.py 文件
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


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
CKPT_DIR = f"../checkpoints/qwen/Mytrain-Qwen3-0.8B-2.8B-{GATING_REFERENCE}" # checkpoint dir
TEST_CHECKPOINT = "0.10000.pth" # test checkpoint
DATA_DIR = "../../washed_data_0520/train" # 这里是我用的 test data，下游任务测试应该不需要。


# 这个是计算专家激活差别的，这个我来算就好了
def _cal_avg_expert_change(data_expert_labels, seq_len, real_lens):
    _, expert_num = data_expert_labels.shape
    data_expert_labels = data_expert_labels.view(-1, seq_len, expert_num)
    intersections = []
    popularities = [0 for _ in range(16)]
    for i, real_len in enumerate(real_lens):
        expert_labels = data_expert_labels[i, :real_len]
        popularity = torch.bincount(expert_labels.view(-1), minlength=16).cpu().numpy()
        for j in range(16):
            popularities[j] += popularity[j]

        intersect = []
        pre = set(expert_labels[0].cpu().numpy())
        for expert_label in expert_labels[1:]:
            cur = set(expert_label.cpu().numpy())
            intersect.append(len(pre.intersection(cur)) / len(pre))
            pre = cur
        intersect = sum(intersect) / len(intersect)
        intersections.append(intersect)
    
    for j in range(len(popularities)):
        popularities[j] /= expert_num * sum(real_lens).item()

    return sum(intersections) / len(intersections), popularities


def _cal_expert_change_budget(expert_labels, seq_len, real_lens, budget, prefill, policy = "fifo"):
    _, expert_num = expert_labels.shape
    expert_labels = expert_labels.view(-1, seq_len, expert_num)
    changes = []
    for i, real_len in enumerate(real_lens):
        change = 0
        if  policy == "fifo":
            to_swapout = 0
        elif policy == "lru":
            lru_cache = np.array([0 for _ in range(budget)])

        inmem_experts = np.array([idx for idx in range(budget)])
        for labels in expert_labels[i, prefill:real_len].cpu().numpy():
            for label in labels:
                if label not in inmem_experts:
                    change += 1
                    if policy == "fifo":
                        inmem_experts[to_swapout] = label
                        to_swapout = (to_swapout + 1) % budget
                    elif policy == "lru":
                        to_swapout = np.argmax(lru_cache)
                        # print(f"need {label}, inmem {inmem_experts}, lru_cache {lru_cache}, swapout {to_swapout}", end = "\r")
                        lru_cache += 1
                        lru_cache[to_swapout] = 0
                        inmem_experts[to_swapout] = label
                    elif policy == "random":
                        to_swapout = random.choice(range(budget))
                        inmem_experts[to_swapout] = label
                else:
                    if policy == "lru":
                        lru_cache += 1
                        label_idx = np.where(inmem_experts == label)[0]
                        lru_cache[label_idx] = 0
                        # print(f"need {label}, inmem {inmem_experts}, lru_cache {lru_cache}", end ="\r")
                    pass
                # breakpoint()
        change = change / (expert_num * (real_len - prefill))
        changes.append(change)
    changes = np.mean(changes)
    return changes


@torch.no_grad()
def prepare_model():
    weights = torch.load(f"{CKPT_DIR}/{TEST_CHECKPOINT}", weights_only = True, map_location="cpu")

    config = AutoConfig.from_pretrained(CONFIG_DIR, trust_remote_code = True)
    config.moe_intermediate_size = MOE_INTERMEDIATE_SIZE
    config.num_experts_per_tok = EXPERT_PER_TOKEN
    config.num_experts = NUM_EXPERTS
    config.gating_reference = GATING_REFERENCE
    config.norm_topk_prob = True
    config.use_moe = USE_MOE

    print("Constructing student model ...")

    model = MyQwen3ForCausalLM(config).to(DEVICE)
    model.eval()
    model.load_state_dict(weights)

    # test_weights = {}
    # for k, v in weights.items():
    #     if k.startswith("model.layers.2.mlp.experts.3."):
    #         test_weights[k.replace("model.layers.2.mlp.experts.3.", "")] = v
    # for i in range(1, 28, 3):
    #     for j in range(0, 16, 5):
    #         t = time.time()
    #         model.model.layers[i].mlp.experts[j].load_state_dict(test_weights)
    #         print(f"Loading one expert weights took {time.time() - t:.5f}s")

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
def forward_step(local_batch_idx, source, target, real_lens, model, token_loss_fn):
    source, target = source.to(DEVICE), target.to(DEVICE)
    prefill_len = 1024
    # position_ids = torch.arange(past_key_values_length, prefill_len, dtype=torch.long, device=DEVICE)
    # position_ids = position_ids.unsqueeze(0).to(DEVICE)

    t = time.time()
    output = model(input_ids = source[:, :prefill_len], output_expert_labels = True, use_cache=True)
    print(f"prefill time: {time.time() - t :.3f}s")
    logits = output.logits
    past_key_values = output.past_key_values

    for idx in range(prefill_len, 1024):
        t = time.time()
        output = model(input_ids = source[:, idx : idx + 1], past_key_values = past_key_values, output_expert_labels = True, use_cache=True)
        print(f"decode time: {time.time() - t :.3f}s")

    # breakpoint()

    changes = [[] for _ in range(len(output.expert_labels))]
    for layer_idx, expert_label in enumerate(output.expert_labels):
        avg_expert_change, popularityes = _cal_avg_expert_change(expert_label, seq_len=SEQ_LEN, real_lens=real_lens)
        print(f"layer {layer_idx} expert intersect: {avg_expert_change * 100 :.3f}, popularity: ", end="")
        for p in popularityes:
            print(f"{p * 100:.2f} ", end = "")
        print("")
        # for policy in ["lru"]:
        #     for budget in [2, 5, 8, 10, 12]:
        #         avg_changes = _cal_expert_change_budget(expert_label, seq_len=SEQ_LEN, real_lens=real_lens.cpu().numpy(), budget=budget, prefill=100, policy=policy)
        #         changes[layer_idx].append(avg_changes)
        #     print(f"layer {layer_idx} {policy} expert change: ", end = "")
        #     for c in changes[layer_idx]:
        #         print(f"{c * 100:.2f} ", end = "")
        #     print()

    # changes = np.mean(changes, axis=0)
    # print(f"avg expert change: ", end = "") 
    # for c in changes:
    #     print(f"{c * 100:.2f} ", end = "")
    print()

    breakpoint()
    loss = token_loss_fn(logits.view(-1, logits.size(-1)), target.reshape(-1))

    print(f"batch: {local_batch_idx}, loss: {loss:.5f}", flush=True)

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
def thread_main():
    print(f"running on DEVICE {DEVICE}")
    torch.set_default_dtype(torch.bfloat16)
    model = prepare_model() # 测试下游任务就只要用这一个函数就好了。


    dataloader = prepare_data()
    token_loss_fn = prepare_loss_optimizer(model)
        
    for local_batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
        loss = forward_step(local_batch_idx, source, target, real_lens, model, token_loss_fn)


if __name__ == "__main__":
    thread_main()
    
