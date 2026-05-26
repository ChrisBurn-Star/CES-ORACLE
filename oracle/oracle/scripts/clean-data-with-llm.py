
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import matplotlib.pyplot as plt
import numpy as np
from old_data_utils import Tokenized_data, Tokenized_data_chat, Tokenized_data_locality, DomainData_DCLM

from torch.utils.data import DataLoader

from datasets import Dataset

import os

import torch




from accelerate import infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


import os
import torch

from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup, AutoModelForCausalLM
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


import numpy as np












plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
plt.rcParams['font.size'] = 15


batch_size = 6


DEVICE = 7
# [0,1,2,3,4,5,6,7]

CUTS = {0:8,1:9,2:10,3:11,4:12,5:13,6:14,7:15} ##31: 8-15
CONFIG_DIR = "/home/jxzhou/PLM_PER/qwen/Qwen3-0.6B" # 就直接用 qwen3 的 dir 就可以


@torch.no_grad()
def process_sample_with_id(x, sample_id, tensorA, M):
    """
    流式处理单个样本：判断是否匹配，并更新 tensorA。

    Args:
        x:         [D] 新样本
        sample_id: 当前样本的唯一标识
        tensorA:   [D, 2] 区间记录矩阵
        M:         至少匹配 M 个方向

    Returns:
        matched_id: 如果命中，返回 sample_id；否则返回 None
    """
    min_vals = tensorA[:, 0]
    max_vals = tensorA[:, 1]

    # 判断落入区间的方向
    in_range_mask = (x >= min_vals) & (x <= max_vals)
    match_count = in_range_mask.sum().item()
    matched = match_count >= M

    # 更新 tensorA
    tensorA[:, 0] = torch.minimum(min_vals, x)
    tensorA[:, 1] = torch.maximum(max_vals, x)

    return sample_id if matched else None, tensorA


def process_batch_and_return_mean(batch, sample_id, tensorA, M=80, K=80):
    """
    批量处理一个 batch（形状 [N, D]）

    Args:
        batch:   [N, D] 批次样本
        tensorA: [D, 2] 区间记录
        M:       每条样本至少要命中 M 个方向
        K:       整个 batch 中至少要有 K 条样本命中

    Returns:
        均值 [D] if 命中样本数 >= K，否则 None
    """
    min_vals = tensorA[:, 0]
    max_vals = tensorA[:, 1]

    # 判断每个样本在每个方向是否在区间内
    in_range_mask = (batch >= min_vals*0.5) & (batch <= max_vals*0.5)  # [N, D]

    # 统计每条样本命中的方向数
    per_sample_match = in_range_mask.sum(dim=1)  # [N]

    # 找出命中的样本（满足 M 个方向）
    matched_mask = per_sample_match >= M        # [N]
    matched_count = matched_mask.sum().item()

    # gengxintensora
    batch_min = batch.min(dim=0).values  # [D]
    batch_max = batch.max(dim=0).values  # [D]
    tensorA[:, 0] = torch.minimum(tensorA[:, 0], batch_min)
    tensorA[:, 1] = torch.maximum(tensorA[:, 1], batch_max)


    # 若命中样本数 ≥ K，返回 batch 的均值
    if matched_count >= K:
        return sample_id, tensorA
    else:
        return None, tensorA


def process_batch_and_return_mean2(batch, sample_id, tensorB, M=80, K=80,rate = 0.8):
    """
    批量处理一个 batch（形状 [N, D]）

    Args:
        batch:   [N, D] 批次样本
        tensorA: [D, 2] 区间记录
        M:       每条样本至少要命中 M 个方向
        K:       整个 batch 中至少要有 K 条样本命中

    Returns:
        均值 [D] if 命中样本数 >= K，否则 None
    """
    min_vals = tensorB[:]


    # 判断每个样本在每个方向是否在区间内
    in_range_mask = (batch.abs() <= min_vals*rate)  # [N, D]

    # 统计每条样本命中的方向数
    per_sample_match = in_range_mask.sum(dim=1)  # [N]

    # 找出命中的样本（满足 M 个方向）
    matched_mask = per_sample_match >= M        # [N]
    matched_count = matched_mask.sum().item()

    # print(matched_count)
# 
    # gengxintensora

    batch_max = batch.abs().max(dim=0).values  # [D]

    tensorB[:] = torch.maximum(tensorB[:], batch_max)


    # 若命中样本数 ≥ K，返回 batch 的均值
    if matched_count >= K:
        return sample_id, tensorB
    else:
        return None, tensorB

def process_batch_and_return_mean3(batch, sample_id, tensorB, M1=80, M2=80, M3=80, alpha=0.1):
    """
    批量处理一个 batch(形状 [N, D])

    Args:
        batch:     [N, D] 批次样本
        sample_id: 批次标识
        tensorB:   List of [K_i] tensors, each storing activation values for dimension i
        M1:        Number of activations a dimension's value must have exactly within alpha to be invalid
        M2:        Number of invalid dimensions for a sample to be invalid
        M3:        Number of invalid samples for the batch to be invalid
        alpha:     Distance threshold for activation comparison

    Returns:
        Tuple (sample_id, tensorB) if batch is invalid (>= M3 invalid samples), else (None, tensorB)
    """
    N, D = batch.shape
    device = batch.device

    # Initialize tensorB as a list of tensors if empty
    if not tensorB:
        tensorB = [torch.tensor([], device=device) for _ in range(D)]

    # Process all samples and dimensions in a vectorized manner
    invalid_dim_counts = torch.zeros(N, device=device, dtype=torch.long)

    for d in range(D):
        values = batch[:, d]  # [N]
        activations = tensorB[d]  # [K_i]

        if len(activations) > 0:
            # Compute distances for all samples in this dimension at once
            # [N, K_i]
            distances = torch.abs(values[:, None] - activations[None, :])
            # Count how many distances are below alpha for each sample
            far_counts = (distances < alpha).sum(dim=1)  # [N]
            # Increment invalid count for samples where far_count == M1
            invalid_dim_counts += (far_counts >= M1).long()
            
            # Update tensorB: add values where far_counts != M1
            mask = (far_counts < M1)  # [N]
            new_activations = values[mask]
            # print(new_activations.shape)
            if new_activations.numel() > 0:
                tensorB[d] = torch.cat([activations, new_activations])
        else:
            # If activations is empty, add all values to tensorB[d]
            tensorB[d] = values.clone()

    # Count invalid samples (where invalid_dim_count >= M2)
    invalid_sample_count = (invalid_dim_counts >= M2).sum().item()
    # if invalid_sample_count>=1010:
    #     print(invalid_sample_count)
    # Check if batch is invalid
    if invalid_sample_count >= M3:
        return sample_id, tensorB
    else:
        return None, tensorB


@torch.no_grad()
def get_small_data_ids(model,test_data):
    model.to(DEVICE)
    model.eval()
    dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=1, shuffle=False)
    O_PROJ = torch.load('/home/jxzhou/PLM_PER/qwen/0614/o_proj_qwen.tensors',map_location='cpu')[1][0].to(DEVICE).clone().detach()
    u, s, V = torch.linalg.svd(O_PROJ, full_matrices=False)
    VS = V.T
    TEM = 900
    tensorA = torch.empty(1024, 2)
    tensorA[:, 0] = float('inf')   # min
    tensorA[:, 1] = float('-inf')  # max
    # tensorB = torch.zeros(TEM)
    tensorB = [torch.tensor([], device=DEVICE) for _ in range(TEM)]

    M1 = 900
    M2 = 1024
    ##1020 M3 30
    SIZE = 100

    IDS = []
    print(f'start finding invaluable data......')
    for i, (input_ids, _, _) in enumerate(dataloader):
        # if i >SIZE:
        #     break
        # print(i)
        datas = input_ids.to(model.device)
        # start_time = time.time()
        output = model(input_ids = datas, output_hidden_states=True, output_attentions=True, labels = datas,output_expert_labels =     False )
        # stop_time = time.time()
        # times = stop_time-start_time
        # print(f"耗时: {times:.4f} 秒")
        # print(datas[0][-10:])
        HIDDEN_STATES = output.attentions[1].clone().detach()##layerindex=20,q_proj_layerindex = 21
        # print(HIDDEN_STATES[1].shape)
        for h in range(len(HIDDEN_STATES)):
            hidden_states = HIDDEN_STATES[h]
            PROJ = hidden_states @ VS
            PROJ = PROJ * s
            PROJ = PROJ.view(-1,1024)

            # matched_id, tensorA = process_batch_and_return_mean(PROJ, sample_id=i, tensorA=tensorA.to(PROJ.device), M=M1,K = M2)
            matched_id, tensorB = process_batch_and_return_mean3(PROJ[:,:TEM], sample_id=i*batch_size+h, tensorB=tensorB, M1=20, M2 = M1,M3 =M2,alpha=0.001)
            
            if matched_id:
                IDS.append(matched_id)
                print(f"invaluable data id: {matched_id} , IDS_len: {len(IDS)}")
            # else:
            #     print(f"{i*batch_size+h}:valuable data")
    torch.save(IDS,f'../ids/DCLM-tested-deleted-data-oracle-ids-{int(len(IDS)/len(dataloader)*100)}-{DEVICE}.pth')
    with open(f"../ids/DCLM-tested-deleted-data-oracle-ids-{int(len(IDS)/len(dataloader)*100)}-{DEVICE}.txt", "w") as f:
        for row in IDS:
            f.write(str(row) + "\n")



        




        







if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained('/home/jxzhou/PLM_PER/qwen/Qwen3-0.6B', device_map="cpu", trust_remote_code=True)
    # test_data = Tokenized_data(tokenizer,CUTS[DEVICE])
    
    test_data = DomainData_DCLM("/home/jxzhou/datasets/dclm/filtered_output", CUTS[DEVICE], 1024, tokenizer)
    
    get_small_data_ids(qwen,test_data)
    