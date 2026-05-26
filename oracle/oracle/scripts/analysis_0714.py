import time
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from utils import DeepSeekDistillation
from models import MyQwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
# from draw_loss import read_pretrain_loss, read_sft_loss

SEQ_LEN = 1024
BATCH_SIZE = 16
NUM_WORKERS = 4
DATA_SHUFFLE = True

USE_MOE = False
MOE_INTERMEDIATE_SIZE = 1536
EXPERT_PER_TOKEN = 2
NUM_EXPERTS = 16
GATING_REFERENCE = "oracle" # oracle / switch

CONFIG_DIR = "/home/jxzhou/PLM_PER/qwen/Qwen3-0.6B"
DATA_DOMAIN = "childstory"
CKPT_DIR = f"../checkpoints/pretrain--0.15000.pth"
DATA_DIR = f"../../data/legal/train" # 2879 * 4096 = 11,792,384
DEVICE = "cuda:7"


@torch.no_grad()
def prepare_model(device) -> MyQwen3ForCausalLM:
    god_model = AutoModelForCausalLM.from_pretrained(CONFIG_DIR, trust_remote_code = True)
    
    config = AutoConfig.from_pretrained(CONFIG_DIR, trust_remote_code = True)
    config.moe_intermediate_size = MOE_INTERMEDIATE_SIZE
    config.num_experts_per_tok = EXPERT_PER_TOKEN
    config.num_experts = NUM_EXPERTS
    config.gating_reference = GATING_REFERENCE
    config.norm_topk_prob = True
    config.use_moe = USE_MOE

    model = MyQwen3ForCausalLM(config).to(device)
    model.load_state_dict(god_model.state_dict())
    # model.load_state_dict(torch.load(CKPT_DIR, weights_only = True, map_location="cpu"))

    print(f'device {device} student model ok, params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B/{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B') # 
    return model


def prepare_data(data_dir):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)
    dataset = DeepSeekDistillation(data_dir, SEQ_LEN, tokenizer)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=DATA_SHUFFLE)

    return dataloader


def prepare_loss_optimizer(s_model):
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=151643, reduction='mean')
    
    return token_loss_fn


@torch.no_grad()
def forward_step(device, batch_idx, source, target, real_len, model, layer_idx):
    # layer_idx: [0, 27]
    source, target = source.to(device), target.to(device)

    model.config.num_hidden_layers = layer_idx
    s_output = model(source, output_hidden_states=False)
    hidden_states = s_output.hidden_states

    decoder_layer : MyQwen3DecoderLayer = model.model.layers[layer_idx]
    hidden_states = decoder_layer.input_layernorm(hidden_states)

    z1s, z1_scaleds, z2s = {}, {}, {}
    for mat in ["q_proj", "k_proj", "v_proj"]:
        if mat == "q_proj":
            weight = decoder_layer.self_attn.q_proj.weight
        elif mat == "k_proj":
            weight = decoder_layer.self_attn.k_proj.weight
        elif mat == "v_proj":
            weight = decoder_layer.self_attn.v_proj.weight
        
        D, d = weight.shape
        u, s, vh = torch.linalg.svd(weight, full_matrices=False) # u = [D, D], s = [D], vh = [d, d]
        v = vh.T  # 转置得到右奇异向量矩阵 V ∈ R^{d x d}

        # 3. 分步变换数据
        # Step 1: 使用右奇异向量变换数据，并乘上奇异值
        z1 = hidden_states @ v                        # [N, d] -> 在 V 基下表示
        z1_scaled = z1 * s                # [N, d] 或者也可以用 diag(S) 矩阵乘法
        
        # Step 2: 使用左奇异向量变换到输出空间
        z2 = z1_scaled @u.T              # [N, D]

        # Step 3: 直接使用原始权重矩阵计算
        Y_direct = hidden_states @ weight.T          # [N, D]

        # 检查两者是否一致
        diff = torch.norm(Y_direct - z2)
        print(f"Difference between direct and step-by-step: {diff.item():.6f}, percentage: {diff.item() / torch.norm(Y_direct):.6%}")

        z1s[mat] = [z1_i[:real_len_i] for z1_i, real_len_i in zip(z1, real_len)]
        z1_scaleds[mat] = [z1_scaled_i[:real_len_i] for z1_scaled_i, real_len_i in zip(z1_scaled, real_len)]
        z2s[mat] = [z2_i[:real_len_i] for z2_i, real_len_i in zip(z2, real_len)]
         
    return z1s, z1_scaleds, z2s


@torch.no_grad()
def redundancy_main():
    model = prepare_model(DEVICE)
    model.eval()
    dataloader = prepare_data(DATA_DIR)
    
    for layer_idx in range(1, 27):
        z1s, z1_scaleds, z2s = {"q_proj":[], "k_proj":[], "v_proj":[]}, {"q_proj":[], "k_proj":[], "v_proj":[]}, {"q_proj":[], "k_proj":[], "v_proj":[]}
        for batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
            t = time.time()
            z1, z1_scaled, z2 = forward_step(DEVICE, batch_idx, source, target, real_lens, model, layer_idx)
            
            for mat in ["q_proj", "k_proj", "v_proj"]:
                z1s[mat].extend(z1[mat])
                z1_scaleds[mat].extend(z1_scaled[mat])
                z2s[mat].extend(z2[mat])
            print(f"batch {batch_idx} time: {time.time() - t:.3f}s", flush=True)

            if batch_idx > 50:
                break

        for mat in ["q_proj", "k_proj", "v_proj"]:
            torch.save(z1s[mat], f"../tensor/redundancy/z1_{mat}_layer{layer_idx}.pt")
            torch.save(z1_scaleds[mat], f"../tensor/redundancy/z1_scaled_{mat}_layer{layer_idx}.pt")
            torch.save(z2s[mat], f"../tensor/redundancy/z2_{mat}_layer{layer_idx}.pt")
        print(f"Saved layer {layer_idx} tensors.")


def draw_embedding_variance(layer_idx, mat, save_dir="../figures/redundancy"):
    z1_load = torch.load(f"../tensor/redundancy/z1_{mat}_layer{layer_idx}.pt", weights_only=True)
    # z1_scaled = torch.load(f"../tensor/redundancy/z1_scaled_{mat}_layer{layer_idx}.pt", weights_only=True)
    z2_load = torch.load(f"../tensor/redundancy/z2_{mat}_layer{layer_idx}.pt", weights_only=True)

    for granularity in ["sample", "token"]:
        if granularity == "sample":
            z1 = torch.concat([z.mean(dim = 0, keepdim = True) for z in z1_load], dim=0)
            # z1_scaled = torch.concat([z.mean(dim = 0, keepdim = True) for z in z1_scaled], dim=0)
            z2 = torch.concat([z.mean(dim = 0, keepdim = True) for z in z2_load], dim=0)
        elif granularity == "token":
            z1 = torch.concat(z1_load, dim=0)
            # z1_scaled = torch.concat(z1_scaled, dim=0)
            z2 = torch.concat(z2_load, dim=0)
        
        for emb_name, embeddings in zip(["z1", "z2"], [z1, z2]):
            # 计算每个样本的方差
            embeddings = embeddings.cpu().numpy()
            for dim in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            # variances = torch.var(embeddings, dim=0).cpu().numpy()
            # 绘制方差分布
                plt.figure(figsize=(10, 6))
                plt.hist(embeddings[:, dim], bins = 30 if granularity == "sample" else 50, color='blue', alpha=0.7)
                plt.title(f'{granularity}-level {emb_name} in Layer {layer_idx} {mat} Dim {dim}')
                plt.xlabel('Variance')
                plt.ylabel('Frequency')
                plt.grid()
                plt.savefig(f"{save_dir}/{granularity}_layer{layer_idx}.{mat}.{emb_name}.dim{dim}.png")
                plt.close()


def draw_embedding_variance_zjx(layer_idx, mat, save_dir="figures/redundancy"):
    z1_load = torch.load(f"redundancy/z1_{mat}_layer{layer_idx}.pt", weights_only=True)
    # z1_scaled = torch.load(f"../tensor/redundancy/z1_scaled_{mat}_layer{layer_idx}.pt", weights_only=True)
    z2_load = torch.load(f"redundancy/z2_{mat}_layer{layer_idx}.pt", weights_only=True)

    for granularity in ["sample", "token"]:
        if granularity == "sample":
            z1 = torch.concat([z.mean(dim = 0, keepdim = True) for z in z1_load], dim=0)
            # z1_scaled = torch.concat([z.mean(dim = 0, keepdim = True) for z in z1_scaled], dim=0)
            z2 = torch.concat([z.mean(dim = 0, keepdim = True) for z in z2_load], dim=0)
        elif granularity == "token":
            z1 = torch.concat(z1_load, dim=0)
            # z1_scaled = torch.concat(z1_scaled, dim=0)
            z2 = torch.concat(z2_load, dim=0)
        # else:
        #     # print(torch.concat(z1_load, dim=0).shape)
        #     # print(z1_load[0].shape)
        #     # print(torch.concat([z[:20] for z in z1_load[:100]], dim=0).shape)
        #     z1 = torch.concat([z[:20] for z in z1_load[:100]], dim=0).view(-1, 4, 1024).mean(dim=1)
        #     # z1_scaled = torch.concat([z.mean(dim = 0, keepdim = True) for z in z1_scaled], dim=0)
        #     z2 = torch.concat([z[:20] for z in z2_load[:100]], dim=0).view(-1, 4, 1024).mean(dim=1)
        
        for emb_name, embeddings in zip(["z1", "z2"], [z1, z2]):
            # 计算每个样本的方差
            embeddings = embeddings.cpu().numpy()
            plt.figure(figsize=(10, 6))
            for d in range(1):
            # for dim in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
                plt.bar([i for i in range(41) ],embeddings[d, :41].T, color='blue', alpha=0.05)
            # plt.scatter([i for i in range(41) ],np.mean(embeddings[:1, :41].T, axis = 1),color = 'red',label = 'average proj')
            # plt.plot([i for i in range(41) ],np.mean(embeddings[:20, :41].T,axis=1), color='blue', alpha=1)
            
            plt.title(f'{granularity}-level {emb_name} in Layer {layer_idx} {mat} Dim0-20')
            plt.xlabel('EVector Index')
            plt.ylabel('Activation')
            plt.legend()
            plt.grid()
            plt.savefig(f"{save_dir}/{granularity}_layer{layer_idx}.{mat}.{emb_name}.png")
            plt.close()

def draw_embedding_variance_main():
    for layer_idx in range(1, 27):
        for mat in ["q_proj", "k_proj", "v_proj"]:
            # draw_embedding_variance(layer_idx, mat, save_dir="../figures/redundancy")
            draw_embedding_variance_zjx(layer_idx, mat, save_dir="figures/redundancy2")

            print(f"Drawn layer {layer_idx} {mat} variance.", flush=True, end="\r")



@torch.no_grad()
def full_forward_step(device, source, target, model,):
    # layer_idx: [0, 27]
    source, target = source.to(device), target.to(device)
    s_output = model(source, output_hidden_states=False)


def time_forawrd_main():
    model = prepare_model(DEVICE)
    model.eval()
    dataloader = prepare_data(DATA_DIR)
    
    for batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
        t = time.time()
        full_forward_step(DEVICE, source, target, model)
        print(f"batch {batch_idx} time: {time.time() - t:.3f}s", flush=True)

        if batch_idx > 50:
            break


if __name__ == "__main__":
    draw_embedding_variance_main()
    # time_forawrd_main()

