import os
import json
import torch
from utils import DeepSeekDistillation

from models import MyQwen3ForCausalLM, My8bitQwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader
from transformer_engine.pytorch import fp8_autocast



LOCAL_BATCH_SIZE =64
SEQ_LEN = 1024
DATA_SHUFFLE = True

USE_MOE = False
MOE_INTERMEDIATE_SIZE = 768
EXPERT_PER_TOKEN = 1
NUM_EXPERTS = 16
GATING_REFERENCE = "oracle"

NUM_WORKERS = 4
DEVICE = 0
MASTER_ADDR = "127.0.0.1"
PORT = "32656"
CONFIG_DIR = "../../Qwen-family/Qwen3-0.6B"
CKPT_DIR = "../checkpoints/qwen/Mytrain-Qwen3-0.6B"
DATA_DIR = "../../washed_data_0520/train" # 2879 * 4096 = 11,792,384

from sklearn.cluster import KMeans
import numpy as np

def cluster_tokens(attn_output_tokens, num_clusters):
    attn_output_tokens = attn_output_tokens.to(torch.float32).cpu().numpy()
    num_tokens = attn_output_tokens.shape[0]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(attn_output_tokens)

    return kmeans

def analyse_cluster(centers: torch.Tensor, tokens: torch.Tensor, labels):
    # use inner product to decide cluster label
    inner_product = torch.matmul(tokens, centers.T)
    ip_label = torch.argmax(inner_product, dim=1)
    # compare if this equal to the results of l2-distance.
    pass

def _cal_avg_expert_change(expert_labels):
    change = []
    pre = set(expert_labels[0].cpu().numpy())
    for i in range(1, len(expert_labels)):
        cur = set(expert_labels[i].cpu().numpy())
        intersect = pre.intersection(cur)
        change.append(len(intersect) / len(pre))
        pre = cur
    return sum(change) / len(change)

def prepare_model():
    config = AutoConfig.from_pretrained(CONFIG_DIR, trust_remote_code = True)
    config.moe_intermediate_size = MOE_INTERMEDIATE_SIZE
    config.num_experts_per_tok = EXPERT_PER_TOKEN
    config.num_experts = NUM_EXPERTS
    config.gating_reference = GATING_REFERENCE
    config.norm_topk_prob = True
    config.use_moe = USE_MOE

    model_class = MyQwen3ForCausalLM
    model = model_class(config).to(DEVICE)
    model.load_state_dict(torch.load(f"{CKPT_DIR}/0.6000.pth"))
    print("Construct student model.")

    print(f'student model ok, params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B/{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B') # 
    return model

def prepare_data():
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)
    dataset = DeepSeekDistillation(DATA_DIR, SEQ_LEN, tokenizer)

    dataloader = DataLoader(dataset, batch_size=LOCAL_BATCH_SIZE, num_workers=NUM_WORKERS)

    return dataloader

@torch.no_grad()
def get_attn_output_wo_residual(source, model):
    source = source.to(DEVICE)
    batch_size, seq_length = source.shape[:2]
    past_key_values_length = 0
    position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=DEVICE)
    position_ids = position_ids.unsqueeze(0).to(DEVICE)

    s_output = model(source, position_ids=position_ids, output_attentions = True)
    attn_output_layers = s_output.attentions
    
    return attn_output_layers


if __name__ == '__main__':
    torch.set_default_dtype(torch.bfloat16)
    dataloader  = prepare_data()
    model = prepare_model()
    
    attn_output_tokens = [[] for layer in range(28)]
    for i, (source, target, real_lens) in enumerate(dataloader, 1):
        attn_outputs = get_attn_output_wo_residual(source, model) # (num_layers, batch_size, seq_length, hidden_size)
        for layer in range(28):
            for data_idx, real_len in enumerate(real_lens):
                attn_output_data = attn_outputs[layer][data_idx, :real_len]
                shuffled_indices = torch.randperm(attn_output_data.size(0))[:real_len // 10]
                attn_output_data = attn_output_data[shuffled_indices]
                attn_output_tokens[layer].append(attn_output_data)
        print(f'Batch {i}')
        if i >= 64:
            break

    for layer_idx, attn_output in enumerate(attn_output_tokens):
        attn_output = torch.concat(attn_output, dim = 0)
        print(f"token count in layer {layer_idx}: {attn_output.shape[0]}")
        kmeans = cluster_tokens(attn_output, num_clusters=NUM_EXPERTS)
        cluster_counts = np.bincount(kmeans.labels_, minlength=NUM_EXPERTS)
        print(f'Cluster counts: {cluster_counts}')

        # save tokens and centers
        torch.save(attn_output, f"../router/mytensors/attn_output_tokens_{layer_idx}.tensor")
        torch.save(torch.tensor(kmeans.cluster_centers_), f"../router/mytensors/centers_{layer_idx}.tensor")

    # cluster_centers = kmeans.cluster_centers_
    # analyse_cluster(torch.tensor(cluster_centers).to(3), torch.tensor(attn_output_tokens).to(3), kmeans.labels_)
    # breakpoint()

