from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from old_data_utils import Tokenized_data

DEVICE = 'cuda:0'
NUM_EXPERTS = 64
HIDDEN_SIZE = 2048
LAYER_NUM = 25
MAT = 'up'
DIR = '../figures/deepseek/'


def draw_data_embedding(test_data, llm):
    hidden_states = [[] for _ in range(LAYER_NUM)]
    with torch.no_grad():
        for i, input_ids in enumerate(test_data):
            input_ids = input_ids.to(llm.device)
            output = llm(input_ids = input_ids, output_hidden_states=True) # output hidden states 就是我要的每层输入
            for layer in range(LAYER_NUM):
                # hidden_states[layer].append(output.hidden_states[layer]) # .reshape(-1, HIDDEN_SIZE)
                hidden_states[layer].append(output.hidden_states[layer].mean(1)) # .reshape(-1, HIDDEN_SIZE)
            # if i > 15:
            #     break

    torch.save(hidden_states, f'{DIR}/ds_sequence_embeddings.tensors')


def subspace_angles_torch(QA, QB):
    QA_H_QB = QA.T @ QB
    sigma = torch.linalg.svdvals(QA_H_QB)
    angle = torch.acos(torch.clamp(sigma, -1., 1.)).mean()
    return angle


def ana_experts(experts, layer):
    u_basis, sing_values, v_basis, Q_left, Q_right = [], [], [], [], []
    for i, expert in enumerate(experts):
        expert = experts[i]
        u, s, v = torch.svd(expert) # [gate_proj, up_proj, down_proj]
        sing_values.append(s[:1408])
        if MAT == 'down':
            u_basis.append(u[:, :1408])
            v_basis.append(v[:1408, :1408])
        else: # up, gate
            u_basis.append(u[:1408])
            v_basis.append(v[:, :1408])
        if i == 0:
            if MAT == 'down':
                experts[i] = u @ torch.diag(s) @ v[:1408].T
            else: # up, gate
                experts[i] = u[:1408] @ torch.diag(s) @ v.T

    torch.save(sing_values, f'{DIR}/ds_{MAT}sing_values_layer_{layer}.tensors')

    # Expert 两两 singular value 的分布 KL divergence
    sv_kl = [[0 for _ in range(NUM_EXPERTS + 1)] for _ in range(NUM_EXPERTS + 1)]
    left_subspace_alignment = [[0 for _ in range(NUM_EXPERTS + 1)] for _ in range(NUM_EXPERTS + 1)]
    right_subspace_alignment = [[0 for _ in range(NUM_EXPERTS + 1)] for _ in range(NUM_EXPERTS + 1)]
    flatten_similarity = [[0 for _ in range(NUM_EXPERTS + 1)] for _ in range(NUM_EXPERTS + 1)]
    norm_diff = [[0 for _ in range(NUM_EXPERTS + 1)] for _ in range(NUM_EXPERTS + 1)]
    for i in range(NUM_EXPERTS + 1):
        t = time.time()
        for j in range(i, NUM_EXPERTS + 1):
            sv_kl[i][j] = torch.nn.functional.kl_div(sing_values[i].log(), sing_values[j], reduction='sum').item()
            left_subspace_alignment[i][j] = subspace_angles_torch(u_basis[i], u_basis[j]).item()
            right_subspace_alignment[i][j] = subspace_angles_torch(v_basis[i], v_basis[j]).item()
            flatten_similarity[i][j] = torch.nn.functional.cosine_similarity(experts[i].flatten(), experts[j].flatten(), dim=0).item()
            norm_diff[i][j] = (torch.norm(experts[i], p=2) - torch.norm(experts[j], p=2)).item()
        print(f"Time for expert {i}: {time.time() - t}")

    np.save(f'{DIR}/ds_{MAT}sing_values_kl_layer_{layer}.tensors', sv_kl)
    draw_heatmap(np.array(sv_kl), f'Singular Value KL-Divergence Layer {layer}', f'{DIR}/ds_{MAT}sing_values_kl_layer_{layer}.png')
    # Expert 两两 左/右奇异向量 的子空间对齐度
    np.save(f'{DIR}/ds_{MAT}left_subspace_alignment_layer_{layer}.tensors', left_subspace_alignment)
    draw_heatmap(np.array(left_subspace_alignment), f'Left Subspace Alignment Layer {layer}', f'{DIR}/ds_{MAT}left_subspace_alignment_layer_{layer}.png')
    np.save(f'{DIR}/ds_{MAT}right_subspace_alignment_layer_{layer}.tensors', right_subspace_alignment)
    draw_heatmap(np.array(right_subspace_alignment), f'Right Subspace Alignment Layer {layer}', f'{DIR}/ds_{MAT}right_subspace_alignment_layer_{layer}.png')
    # 专家 flatten 后两两余弦相似度
    np.save(f'{DIR}/ds_{MAT}flatten_similarity_layer_{layer}.tensors',flatten_similarity)
    draw_heatmap(np.array(flatten_similarity), f'Flatten Similarity Layer {layer}', f'{DIR}/ds_{MAT}flatten_similarity_layer_{layer}.png')
    # 专家两两范数差异
    np.save(f'{DIR}/ds_{MAT}norm_diff_layer_{layer}.tensors', norm_diff)
    draw_heatmap(np.array(norm_diff), f'Norm Difference Layer {layer}', f'{DIR}/ds_{MAT}norm_diff_layer_{layer}.png')


def draw_heatmap(data, title):
    plt.close()
    plt.imshow(data, cmap = 'hot', interpolation = 'nearest')
    plt.colorbar()
    plt.title(title)
    plt.savefig(f'{DIR}/{title}.png')


def pairwise_distances(x):
    # 展开x以准备进行广播相加
    xx = (x ** 2).sum(dim=1).view(-1, 1)
    xy = torch.mm(x, x.t())    # 计算点积并展开    
    distance_matrix = xx - 2 * xy + xx.t()    # 计算距离矩阵    
    # 因为可能有浮点数精度问题导致负数出现，所以这里需要将负数变成0
    distance_matrix = torch.clamp(distance_matrix, min=0)
    # 开方得到真实欧氏距离
    return torch.sqrt(distance_matrix)


def ana_emb_distance():
    hidden_states = torch.load(f'{DIR}/ds_token_embeddings.tensors', weights_only=True)
    # breakpoint()
    for layer in range(LAYER_NUM):
        # hidden_states[layer] = torch.cat(hidden_states[layer], dim = 0)
        token_embed = hidden_states[layer][0][0]

        dist = token_embed @ token_embed.T # pairwise_distances(hidden_states[layer])
        # dist = pairwise_distances(token_embed)

        dist = dist.clip(0, dist.max() * 1e-4) #/ dist.max()
        draw_heatmap(dist.float().cpu().numpy(), f'deepseek_token_inner_product_layer{layer}')


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    # tokenizer = AutoTokenizer.from_pretrained("../DeepSeek-16B-2.8B", trust_remote_code=True)
    # deepseek = AutoModelForCausalLM.from_pretrained('../DeepSeek-16B-2.8B', device_map="auto", trust_remote_code=True)
    # param_num=  sum(p.numel() for p in deepseek.parameters())

    ana_emb_distance()
    # for layer in range(1, 27):
    #     shared_expert = deepseek.model.layers[layer].mlp.shared_experts
    #     experts = deepseek.model.layers[layer].mlp.experts
    #     breakpoint()
    #     experts.insert(0, shared_expert)
    #     experts = [expert.up_proj.weight.float() for expert in experts] # [gate_proj, up_proj, down_proj]
    #     with torch.no_grad():
    #         ana_experts(experts, layer)

    # test_data = Tokenized_data(['openweb', 'legal', 'med'], tokenizer, total=50, is_test=True)
    # draw_data_embedding(test_data, deepseek)
