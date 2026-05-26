from cProfile import label
from tkinter import Label
from seaborn import heatmap
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from old_data_utils import Tokenized_data, Tokenized_data_chat, Tokenized_data_locality
from sklearn.decomposition import PCA
import umap
from sklearn.metrics.pairwise import cosine_similarity
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from utils import Benchmark_ultrachat_FewShot_Qwen3, Benchmark_Mag_FewShot_Qwen3,Benchmark_TriviaQA_FewShot_Qwen3,Benchmark_XSUM_FewShot_Qwen3,Benchmark_GPQA_Qwen3,Benchmark_DROP_FewShot_Qwen3,compute_exact,compute_f1,Benchmark_BBH_FewShot_Qwen3,Benchmark_MMLU_FewShot_Qwen3,Benchmark_DROP, Benchmark_MMLU, Benchmark_GPQA,Benchmark_BBH,Benchmark_MMLU_FewShot,Benchmark_BBH_FewShot,Benchmark_XSUM_FewShot
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import re
from datasets import Dataset

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
from collections import Counter


from transformers import LogitsProcessor, LogitsProcessorList
import os
import torch
import torch.nn as nn

from utils import DeepSeekDistillation
from models import MyQwen3ForCausalLM # 这里还需要 models/myqwen.py, qwen3config.py 文件
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup, AutoModelForCausalLM
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
# from transformer_engine.pytorch import fp8_autocast
from accelerate import dispatch_model
from accelerate import Accelerator
import re
import string
from collections import Counter

from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm

from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr









plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
plt.rcParams['font.size'] = 15



# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,6,7'

BATCH_SIZE = 8
SEQ_LEN = 2048
EXPERT_NUM = 64
TOPK_EXPERT = 6
T_DTYPE = torch.bfloat16
S_DTYPE = torch.float32
GATING_REFERENCE = 'attn_output'
LLM_DIR = "/home/jxzhou/PLM_PER/qwen/DeepSeek-16B-2.8B"




TASK = 'CP'
NEW_TOKENS = 200  ####MMLU 3; BBH 300 XSUM 30
PAD = 5 #####MMLU:5 BBH:5 XSUM:0
keyword = "Answer:"  ###MMLU
# keyword = "!"  ###MMLU

# keyword = "So the answer is" ### BBH
# keyword = "Summary"  ###XSUM
DATASET = {'MMLU': Benchmark_MMLU_FewShot_Qwen3,'BBH':Benchmark_BBH_FewShot_Qwen3, 'XSUM/processed':Benchmark_XSUM_FewShot, 'DROP':Benchmark_DROP_FewShot_Qwen3, 'GPQA':Benchmark_GPQA_Qwen3,'XSUM':Benchmark_XSUM_FewShot_Qwen3,'TRIVIAQA':Benchmark_TriviaQA_FewShot_Qwen3,'MAG':Benchmark_Mag_FewShot_Qwen3, 'ULTRACHAT':Benchmark_ultrachat_FewShot_Qwen3}

if TASK == 'MMLU' or TASK == 'BBH':
    base_dir = f'/home/jxzhou/PLM_PER/datasets/{TASK}'
    subset_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    result_dict = {}





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



# MMLU: mmlu-oracle-0.xxx.pth
# 10: 10% 的时候能输出正确的答案格式。
# 30: 50% 的情况下能输出正确的答案格式。
# 50: 80% 的情况下能输出正确的答案格式。
# 80: 90% 的情况下能输出正确的答案格式。
# 500: 100% 的情况下能输出正确的答案格式。


steps = 0
MODEL_NAME = 'cp_' + GATING_REFERENCE + str(steps)


DEVICE = 4
CONFIG_DIR = "/home/jxzhou/PLM_PER/qwen/Qwen3-0.6B" # 就直接用 qwen3 的 dir 就可以
# CKPT_DIR = "/home/fdong/lowmem_qwen/checkpoints" # checkpoint dir
CKPT_DIR = "/home/fdong/lowmem_qwen/checkpoints" # checkpoint dir


TEST_CHECKPOINT = f"pretrain-oracle-0.{steps}.pth" # test checkpoint
# TEST_CHECKPOINT = "switch.0.13000.pth" # test checkpoint


@torch.no_grad()
def prepare_qwenmodel():
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

    # print(f'rank {DEVICE} student model ok, params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B/{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B') # 
    # if os.path.exists(f"{CKPT_DIR}/{TEST_CHECKPOINT}"):
    #     model.load_state_dict(torch.load(f"{CKPT_DIR}/{TEST_CHECKPOINT}", weights_only = True, map_location="cpu"))
    # else:
    #     print('no ckpts!')



    return model










def analysis(test_data, model,name, sub):
    dataloader = DataLoader(test_data, batch_size=1, num_workers=1, shuffle=False)
    hiddens = [[] for i in range(28)]
    routing = [[] for i in range(28)]
    inputs = []
    for i, input_ids in enumerate(dataloader):
        if i >2:
            break
        print(i)
        datas = input_ids["input_ids"].to(model.device)
        output = model(input_ids = datas, output_hidden_states=True, output_attentions=True, labels = datas,output_expert_labels = True )
        # print(datas[0][-10:])
        inputs.append(datas.view(-1))
        print(len(output.hidden_states))
        for layer in range(28):
            hiddens[layer].append(output.hidden_states[layer+1])
            routing[layer].append(output.expert_labels[layer])

    torch.save(hiddens, f'/home/jxzhou/PLM_PER/qwen/0614/hiddenstates_{sub}_{name}.tensors')
    torch.save(inputs, f'/home/jxzhou/PLM_PER/qwen/0614/inputs_{sub}_{name}.tensors')
    torch.save(routing, f'/home/jxzhou/PLM_PER/qwen/0614/routing_{sub}_{name}.tensors')




def get_experts(model,name,TASK):
    EXPERTS = [[] for l in range(28)]
    for layer in range(28):
        if name == 'qwen':
            # EXPERTS[layer].append(model.model.layers[layer].mlp.up_proj.weight.data)
            EXPERTS[layer].append(model.model.layers[layer].self_attn.q_proj.weight.data)

            
        else:
            for e in range(16):
                EXPERTS[layer].append(model.model.layers[layer].mlp.experts[e].up_proj.weight.data)
    torch.save(EXPERTS, f'/home/jxzhou/PLM_PER/qwen/0614/q_proj_{name}.tensors')
    print(EXPERTS[0][0].shape)

def visualize_projection_distribution(parameter_matrix, data_vector, indecies , top_k, save_path,layer):
    """
    对参数矩阵进行SVD，使用前K个特征方向，计算数据向量在这些方向上的投影分布，并绘图保存。

    参数:
        parameter_matrix (np.ndarray): 参数矩阵，形状为 [M, N]
        data_vector (np.ndarray): 单个数据向量，形状为 [1, N]
        top_k (int): 使用前K个奇异向量方向
        save_path (str): 图像保存路径
    """
    # 创建保存目录（如果不存在）
    output_dir = os.path.dirname(f'0614_pngs/{save_path}')
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if len(parameter_matrix)>1:
        parameter_matrix = torch.cat(parameter_matrix,dim = 0).float().cpu().numpy()
    else:
        parameter_matrix = parameter_matrix[0].float().cpu().numpy()

    data_vector_example = data_vector[0][0][indecies].float().detach().cpu().numpy()[1]
    data_vector_instruct = data_vector[0][0][indecies].float().detach().cpu().numpy()[3]
    
    # print(data_vector.shape)
    
    # 执行SVD分解
    _, _, Vt = np.linalg.svd(parameter_matrix, full_matrices=False)

    # 取前K个特征方向（右奇异向量）
    top_k_directions = Vt[:top_k, :]  # shape: [K, N]

    # 数据向量在这些方向上的投影
    projections1 = np.dot(top_k_directions, data_vector_example.T).flatten()  # shape: [K]

    # 可视化投影分布
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, top_k + 1), projections1)
    plt.xlabel("Top-K Singular Directions")
    plt.ylabel("Projection Magnitude")
    plt.title("Projection of Data Vector onto Top-K Singular Directions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'0612_pngs/{layer}_{save_path}_example.png')
    plt.close()

    projections2 = np.dot(top_k_directions, data_vector_instruct.T).flatten()  # shape: [K]

    # 可视化投影分布
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, top_k + 1), projections2)
    plt.xlabel("Top-K Singular Directions")
    plt.ylabel("Projection Magnitude")
    plt.title("Projection of Data Vector onto Top-K Singular Directions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'0612_pngs/{layer}_{save_path}_instruct.png')
    plt.close()

def visualize_cluster_distribution(data_vector, indecies, save_path,layer):
    """
    对参数矩阵进行SVD，使用前K个特征方向，计算数据向量在这些方向上的投影分布，并绘图保存。

    参数:
        parameter_matrix (np.ndarray): 参数矩阵，形状为 [M, N]
        data_vector (np.ndarray): 单个数据向量，形状为 [1, N]
        top_k (int): 使用前K个奇异向量方向
        save_path (str): 图像保存路径
    """
    # 创建保存目录（如果不存在）
    output_dir = os.path.dirname(f'0612_pngs/{save_path}')
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    data_vector_all = data_vector[0][0].float().detach().cpu().numpy()
    # data_vector_example = data_vector[0][0][indecies].float().detach().cpu().numpy()[1]
    # data_vector_instruct = data_vector[0][0][indecies].float().detach().cpu().numpy()[3]
    
    # print(data_vector.shape)
    
    u = umap.UMAP(n_components=2).fit(data_vector_all)

    data_vector_all = u.transform(data_vector_all)
    data_vector_example = data_vector_all[indecies.cpu().numpy()]
    data_vector_instruct = data_vector_all[indecies.cpu().numpy()]
    # print(data_vector_all[indecies.cpu().numpy()].shape)



    # 可视化投影分布
    plt.figure(figsize=(8, 5))
    plt.scatter(data_vector_all[:,0],data_vector_all[:,1],label = 'all tokens',alpha = 0.2)
    plt.scatter(data_vector_example[1,0],data_vector_example[1,1],label = 'instruction in example')
    plt.scatter(data_vector_instruct[3,0],data_vector_instruct[3,1],label = 'instruction in question')

    plt.xlabel("umap dim 1")
    plt.ylabel("umap dim 2")
    plt.title("umap cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'0612_pngs/{layer}_{save_path}_cluster.png')
    plt.close()

def show_distribution_relation_steps(STEPS,TASK):
    RELATION = [[] for i in range(len(STEPS))]
    output_dir = os.path.dirname('0614_pngs/show_distribution_relation_steps.png')
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for i in range(len(STEPS)):
        if STEPS[i] == 'God':
            parameter_matrix = torch.load('/home/jxzhou/PLM_PER/qwen/0612/experts_qwen.tensors')

            data_vector = torch.load(f'/home/jxzhou/PLM_PER/qwen/0612/hiddenstates_{TASK}_qwen.tensors')
            sft_oracle_inputs = torch.load(f'/home/jxzhou/PLM_PER/qwen/0612/inputs_{TASK}_qwen.tensors')
            indecies = (sft_oracle_inputs[0] == 25).nonzero(as_tuple=False).squeeze()
            # print(sft_oracle_inputs)
            # print(len(indecies))
            

            
            for layer in range(0,28,5):
                print(f'{i}, {layer}')
                if len(parameter_matrix)>1:
                    parameter_matrix0 = torch.cat(parameter_matrix[layer],dim = 0).float().cpu().numpy()
                else:
                    parameter_matrix0 = parameter_matrix[layer][0].float().cpu().numpy()


                _, _, Vt = np.linalg.svd(parameter_matrix0, full_matrices=False)

                #` 取前K个特征方向（右奇异向量）
                top_k_directions = Vt[:20, :]  # shape: [K, N]`
                data_vector_example = data_vector[layer][0][0][indecies].float().detach().cpu().numpy()[1]
                data_vector_instruct = data_vector[layer][0][0][indecies].float().detach().cpu().numpy()[-1]
                projections1 = np.dot(top_k_directions, data_vector_example.T).flatten()
                projections2 = np.dot(top_k_directions, data_vector_instruct.T).flatten()

                distance = np.sum(np.abs(projections1-projections2))
                # cosine_re= np.exp(-euclidean(projections1, projections2))
                RELATION[i].append(distance)
        else:
            parameter_matrix = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_sft_oracle{STEPS[i]}.tensors')
            data_vector = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/hiddenstates_{TASK}_sft_oracle{STEPS[i]}.tensors')
            sft_oracle_inputs = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/inputs_{TASK}_sft_oracle{STEPS[i]}.tensors')
            indecies = (sft_oracle_inputs[0] == 25).nonzero(as_tuple=False).squeeze()
            # print(sft_oracle_inputs)
            # print(len(indecies))
            

            
            for layer in range(0,28,5):
                print(f'{i}, {layer}')
                if len(parameter_matrix)>1:
                    parameter_matrix0 = torch.cat(parameter_matrix[layer],dim = 0).float().cpu().numpy()
                else:
                    parameter_matrix0 = parameter_matrix[layer][0].float().cpu().numpy()


                _, _, Vt = np.linalg.svd(parameter_matrix0, full_matrices=False)

                #` 取前K个特征方向（右奇异向量）
                top_k_directions = Vt[:20, :]  # shape: [K, N]`
                data_vector_example = data_vector[layer][0][0][indecies].float().detach().cpu().numpy()[1]
                data_vector_instruct = data_vector[layer][0][0][indecies].float().detach().cpu().numpy()[-1]
                projections1 = np.dot(top_k_directions, data_vector_example.T).flatten()
                projections2 = np.dot(top_k_directions, data_vector_instruct.T).flatten()

                distance = np.sum(np.abs(projections1-projections2))
                # cosine_re= np.exp(-euclidean(projections1, projections2))
                RELATION[i].append(distance)
    plt.figure(figsize=(20, 8))
    for s in range(len(STEPS)):
        plt.plot([i for i in range(0,28,5)], RELATION[s], label = f'ckpt-{STEPS[s]}')
    plt.legend()
    plt.ylabel('distribution distance')
    plt.xlabel('layers')
    plt.title('Distribution Similarity')
    plt.savefig(f'0614_pngs/show_distribution_relation_steps_{TASK}.png')
    plt.close()


def show_distribution_steps(STEPS,TASK):
    # RELATION = [[] for i in range(len(STEPS))]
    DISTRIBUTION = [[] for i in range(len(STEPS))]
    output_dir = os.path.dirname('0619_pngs/show_distribution_steps.png')
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    test_token_id = 10
    expert_part = 'gate'
    for i in range(len(STEPS)):
        if STEPS[i] == 'God':
            parameter_matrix = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_qwen_{expert_part}.tensors')

            data_vector = torch.load(f'/home/jxzhou/PLM_PER/qwen/0612/hiddenstates_{TASK}_qwen.tensors')
            sft_oracle_inputs = torch.load(f'/home/jxzhou/PLM_PER/qwen/0612/inputs_{TASK}_qwen.tensors')
            indecies = (sft_oracle_inputs[0] == 25).nonzero(as_tuple=False).squeeze()
            
            for layer in range(0,28,5):
                print(f'{i}, {layer}')
                if len(parameter_matrix)>1:
                    parameter_matrix0 = torch.cat(parameter_matrix[layer],dim = 0).float().cpu().numpy()
                else:
                    parameter_matrix0 = parameter_matrix[layer][0].float().cpu().numpy()


                U, _, Vt = np.linalg.svd(parameter_matrix0, full_matrices=False)

                #` 取前K个特征方向（右奇异向量）
                # top_k_directions = Vt[:20, :]  # shape: [K, N]`
                # print(Vt.shape)
                top_k_directions = Vt[:20, :]  # shape: [K, N]`
                # print(U.shape,Vt.shape)
                # top_k_directions = U.T[:20, :]  # shape: [K, N]`


        

                
                # data_vector_example = data_vector[layer][0][0][indecies].float().detach().cpu().numpy()[1]
                # projections1 = np.dot(top_k_directions, data_vector_example.T).flatten()
                # projections2 = np.dot(top_k_directions, data_vector_instruct.T).flatten()

                data_vector_token = data_vector[layer][0][0][indecies].float().detach().cpu().numpy()[-1]

                # data_vector_token = data_vector[layer][0][0].float().detach().cpu().numpy()[test_token_id]
                projections = np.dot(top_k_directions, data_vector_token.T).flatten()

                DISTRIBUTION[i].append(projections)

                # distance = np.sum(np.abs(projections1-projections2))
                # cosine_re= np.exp(-euclidean(projections1, projections2))
                # RELATION[i].append(distance)
        else:
            parameter_matrix = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_sft_oracle{STEPS[i]}_{expert_part}.tensors')
            data_vector = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/hiddenstates_{TASK}_sft_oracle{STEPS[i]}.tensors')
            sft_oracle_inputs = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/inputs_{TASK}_sft_oracle{STEPS[i]}.tensors')
            indecies = (sft_oracle_inputs[0] == 25).nonzero(as_tuple=False).squeeze()

            
            
            for layer in range(0,28,5):
                print(f'{i}, {layer}')
                if len(parameter_matrix)>1:
                    parameter_matrix0 = torch.cat(parameter_matrix[layer],dim = 0).float().cpu().numpy()
                else:
                    parameter_matrix0 = parameter_matrix[layer][0].float().cpu().numpy()


                U, _, Vt = np.linalg.svd(parameter_matrix0, full_matrices=False)

                #` 取前K个特征方向（右奇异向量）
                # top_k_directions = Vt[:20, :]  # shape: [K, N]`
                # print(Vt.shape)
                top_k_directions = Vt[:20, :]  # shape: [K, N]`
                # print(U.shape,Vt.shape)
                # top_k_directions = U.T[:20, :]  # shape: [K, N]`
                # data_vector_example = data_vector[layer][0][0][indecies].float().detach().cpu().numpy()[1]
                # projections1 = np.dot(top_k_directions, data_vector_example.T).flatten()
                # projections2 = np.dot(top_k_directions, data_vector_instruct.T).flatten()

                data_vector_token = data_vector[layer][0][0][indecies].float().detach().cpu().numpy()[-1]

                # data_vector_token = data_vector[layer][0][0].float().detach().cpu().numpy()[test_token_id]
                projections = np.dot(top_k_directions, data_vector_token.T).flatten()

                DISTRIBUTION[i].append(projections)

                # distance = np.sum(np.abs(projections1-projections2))
                # cosine_re= np.exp(-euclidean(projections1, projections2))
                # RELATION[i].append(distance)
    for layer in range(len(DISTRIBUTION[0])):
        plt.figure(figsize=(20, 8))
        for s in range(len(STEPS)):
            plt.plot([i for i in range(len(DISTRIBUTION[s][layer]))], DISTRIBUTION[s][layer], label = f'ckpt-{STEPS[s]}')
        plt.legend()
        plt.ylabel('projections')
        plt.xlabel('top-k singular directions')
        plt.title('Distribution')
        plt.savefig(f'0619_pngs/show_distribution_steps_{TASK}_{expert_part}_layer{0+layer*5}.png')
        plt.close()





def show_distribution_activate_expert_steps(STEPS,TASK):
    # RELATION = [[] for i in range(len(STEPS))]
    DISTRIBUTION = [[] for i in range(len(STEPS))]
    output_dir = os.path.dirname('0619_pngs/show_distribution_steps.png')
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    test_token_id = 10
    expert_part = 'gate'
    activated_experts = [[]for j in range(28)]
    for l in range(28):
        for step in range(len(STEPS)):
            activated_expert = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/routing_{TASK}_sft_oracle{STEPS[step]}.tensors')
            activated_experts[l].append(activated_expert[l])
    for i in range(len(STEPS)):
        if STEPS[i] == 'God':
            parameter_matrix = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_qwen_{expert_part}.tensors')

            data_vector = torch.load(f'/home/jxzhou/PLM_PER/qwen/0612/hiddenstates_{TASK}_qwen.tensors')
            sft_oracle_inputs = torch.load(f'/home/jxzhou/PLM_PER/qwen/0612/inputs_{TASK}_qwen.tensors')
            indecies = (sft_oracle_inputs[0] == 25).nonzero(as_tuple=False).squeeze()
            
            for layer in range(0,28,5):
                print(f'{i}, {layer}')
                if len(parameter_matrix)>1:
                    # parameter_matrix0 = torch.cat(parameter_matrix[layer],dim = 0).float().cpu().numpy()
                    parameter_matrix0 = torch.cat(parameter_matrix[layer][activated_experts[layer][i]],dim = 0).float().cpu().numpy() ## activated expert
                    # parameter_matrix0 = torch.cat(parameter_matrix[layer][torch.cat(activated_experts[layer], dim = 0)], dim = 0).float().cpu().numpy() ## activated experts
                else:
                    parameter_matrix0 = parameter_matrix[layer][0].float().cpu().numpy()


                U, _, Vt = np.linalg.svd(parameter_matrix0, full_matrices=False)

                #` 取前K个特征方向（右奇异向量）
                # top_k_directions = Vt[:20, :]  # shape: [K, N]`
                # print(Vt.shape)
                top_k_directions = Vt[:20, :]  # shape: [K, N]`
                # print(U.shape,Vt.shape)
                # top_k_directions = U.T[:20, :]  # shape: [K, N]`


        

                
                # data_vector_example = data_vector[layer][0][0][indecies].float().detach().cpu().numpy()[1]
                # projections1 = np.dot(top_k_directions, data_vector_example.T).flatten()
                # projections2 = np.dot(top_k_directions, data_vector_instruct.T).flatten()

                data_vector_token = data_vector[layer][0][0][indecies].float().detach().cpu().numpy()[-1]

                # data_vector_token = data_vector[layer][0][0].float().detach().cpu().numpy()[test_token_id]
                projections = np.dot(top_k_directions, data_vector_token.T).flatten()

                DISTRIBUTION[i].append(projections)

                # distance = np.sum(np.abs(projections1-projections2))
                # cosine_re= np.exp(-euclidean(projections1, projections2))
                # RELATION[i].append(distance)
        else:
            parameter_matrix = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_sft_oracle{STEPS[i]}_{expert_part}.tensors')
            data_vector = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/hiddenstates_{TASK}_sft_oracle{STEPS[i]}.tensors')
            sft_oracle_inputs = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/inputs_{TASK}_sft_oracle{STEPS[i]}.tensors')
            indecies = (sft_oracle_inputs[0] == 25).nonzero(as_tuple=False).squeeze()

            
            
            for layer in range(0,28,5):
                print(f'{i}, {layer}')
                if len(parameter_matrix)>1:
                    # parameter_matrix0 = torch.cat(parameter_matrix[layer],dim = 0).float().cpu().numpy()


                    parameter_matrix0 = []
                    for e in activated_experts[layer][i][0][indecies[-1]]:
                        print(e)
                        parameter_matrix0.append(parameter_matrix[layer][e])
                    # print(len(parameter_matrix0))
                    parameter_matrix0 = torch.cat(parameter_matrix0, dim = 0).float().cpu().numpy() 

                    # parameter_matrix0 = torch.cat(torch.tensor(parameter_matrix[layer])[activated_experts[layer][i][0][indecies[-1]]],dim = 0).float().cpu().numpy() ## activated expert
                    # parameter_matrix0 = torch.cat(torch.tensor(parameter_matrix[layer])[torch.cat(activated_experts[layer], dim = 0)], dim = 0).float().cpu().numpy() ## activated experts
                else:
                    parameter_matrix0 = parameter_matrix[layer][0].float().cpu().numpy()


                U, _, Vt = np.linalg.svd(parameter_matrix0, full_matrices=False)

                #` 取前K个特征方向（右奇异向量）
                # top_k_directions = Vt[:20, :]  # shape: [K, N]`
                # print(Vt.shape)
                top_k_directions = Vt[:20, :]  # shape: [K, N]`
                # print(U.shape,Vt.shape)
                # top_k_directions = U.T[:20, :]  # shape: [K, N]`
                # data_vector_example = data_vector[layer][0][0][indecies].float().detach().cpu().numpy()[1]
                # projections1 = np.dot(top_k_directions, data_vector_example.T).flatten()
                # projections2 = np.dot(top_k_directions, data_vector_instruct.T).flatten()

                data_vector_token = data_vector[layer][0][0][indecies].float().detach().cpu().numpy()[-1]

                # data_vector_token = data_vector[layer][0][0].float().detach().cpu().numpy()[test_token_id]
                projections = np.dot(top_k_directions, data_vector_token.T).flatten()

                DISTRIBUTION[i].append(projections)

                # distance = np.sum(np.abs(projections1-projections2))
                # cosine_re= np.exp(-euclidean(projections1, projections2))
                # RELATION[i].append(distance)
    for layer in range(len(DISTRIBUTION[0])):
        plt.figure(figsize=(20, 8))
        for s in range(len(STEPS)):
            plt.plot([i for i in range(len(DISTRIBUTION[s][layer]))], DISTRIBUTION[s][layer], label = f'ckpt-{STEPS[s]}')
        plt.legend()
        plt.ylabel('projections')
        plt.xlabel('top-k singular directions')
        plt.title('Distribution')
        plt.savefig(f'0619_pngs/show_distribution_activated_experts_steps_{TASK}_{expert_part}_layer{0+layer*5}.png')
        plt.close()



def show_distribution_count_steps(STEPS,TASK):
    # RELATION = [[] for i in range(len(STEPS))]
    DISTRIBUTION = [[] for i in range(len(STEPS))]
    output_dir = os.path.dirname('0623_pngs/show_distribution_steps.png')
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    test_token_id = 10
    expert_part = 'up'
    PS = 8

    r = 0.05
    for i in range(len(STEPS)):
        if STEPS[i] == 'God':
            parameter_matrix = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_qwen.tensors')

            # data_vector = torch.load(f'/home/jxzhou/PLM_PER/qwen/0612/hiddenstates_{TASK}_qwen.tensors')
            # sft_oracle_inputs = torch.load(f'/home/jxzhou/PLM_PER/qwen/0612/inputs_{TASK}_qwen.tensors')
            # indecies = (sft_oracle_inputs[0] == 25).nonzero(as_tuple=False).squeeze()
            
            for layer in range(0,28,5):
                print(f'{i}, {layer}')
                if len(parameter_matrix)>1:
                    parameter_matrix0 = torch.cat(parameter_matrix[layer],dim = 0).float().cpu().numpy()
                else:
                    parameter_matrix0 = parameter_matrix[layer][0].float().cpu().numpy()


                U, S, Vt = np.linalg.svd(parameter_matrix0, full_matrices=False)

                max_val = int(np.ceil(np.max(S)))
                bins = np.arange(4.5, PS, r)

                # 统计各 bin 中的频数（归一化为比例）
                hist, _ = np.histogram(S, bins=bins)
                proportions = hist / np.sum(hist)  # 转换为比例

                DISTRIBUTION[i].append(proportions)

        else:
            if STEPS[i] == 0 and TASK != 'CP':
                parameter_matrix = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_cp_oracle18000_CP_{expert_part}.tensors')
            else:
                parameter_matrix = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_cp_oracle{STEPS[i]}_{TASK}_{expert_part}.tensors')
            # data_vector = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/hiddenstates_{TASK}_cp_oracle{STEPS[i]}.tensors')
            # sft_oracle_inputs = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/inputs_{TASK}_cp_oracle{STEPS[i]}.tensors')
            # indecies = (sft_oracle_inputs[0] == 25).nonzero(as_tuple=False).squeeze()

            
            
            for layer in range(0,28,5):
                print(f'{i}, {layer}')
                if len(parameter_matrix)>1:
                    parameter_matrix0 = torch.cat(parameter_matrix[layer],dim = 0).float().cpu().numpy()
                else:
                    parameter_matrix0 = parameter_matrix[layer][0].float().cpu().numpy()


                U, S, Vt = np.linalg.svd(parameter_matrix0, full_matrices=False)
                # S = np.log(S)
                

                max_val = int(np.ceil(np.max(S)))
                bins = np.arange(2.5, PS, r)

                # 统计各 bin 中的频数（归一化为比例）
                hist, _ = np.histogram(S, bins=bins)
                proportions = hist / np.sum(hist)  # 转换为比例

                DISTRIBUTION[i].append(proportions)
        print(S)
    for s in range(len(STEPS)):
        plt.figure(figsize=(20, 20))
        for layer in range(len(DISTRIBUTION[0])):
            plt.plot(bins[:-1], np.log(DISTRIBUTION[s][layer]), label = f'layer-{layer}',c = 'b',alpha = (layer+1)*0.03)
        plt.legend()
        plt.ylabel('proportion')
        plt.xlabel('singular values')
        plt.title('Singularity Distribution')
        plt.savefig(f'0623_pngs/show_distribution_count_steps{STEPS[s]}_{TASK}_{expert_part}_layer{0+layer*5}.png')
        plt.close()


def show_across_steps(STEPS,TASK):
    # RELATION = [[] for i in range(len(STEPS))]
    DISTRIBUTION = [[[] for l in range(28)] for k in range(1200)]
    output_dir = os.path.dirname('0623_pngs/show_distribution_steps.png')
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    expert_part = 'gate'
    expert_id = 'all'
    # for expert_id in range(16):
    for k in range(100,1200,200):
        for i in range(1,len(STEPS)):

            parameter_matrix1 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_cp_oracle{STEPS[i-1]}_{TASK}_{expert_part}.tensors')
            parameter_matrix2 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_cp_oracle{STEPS[i]}_{TASK}_{expert_part}.tensors')
            for layer in range(0,28,5):
                print(f'{k}, {i}, {layer}')
                if len(parameter_matrix1)>1:
                    parameter_matrix01 = torch.cat(parameter_matrix1[layer],dim = 0).float().cpu().numpy()
                    # parameter_matrix01 = parameter_matrix1[layer][expert_id].float().cpu().numpy()

                else:
                    parameter_matrix01 = parameter_matrix1[layer][0].float().cpu().numpy()


                if len(parameter_matrix2)>1:
                    parameter_matrix02 = torch.cat(parameter_matrix2[layer],dim = 0).float().cpu().numpy()
                    # parameter_matrix02 = parameter_matrix2[layer][expert_id].float().cpu().numpy()
                    
                else:
                    parameter_matrix02 = parameter_matrix2[layer][0].float().cpu().numpy()


                U, S, Vt1 = np.linalg.svd(parameter_matrix01, full_matrices=False)
                U, S, Vt2 = np.linalg.svd(parameter_matrix02, full_matrices=False)
                # print(Vt1.shape)
                # S = np.log(S)
                Vt1 = Vt1[-k:]
                Vt2 = Vt2[-k:]


                # M = Vt1.T @ Vt2
                # _, s, _ = np.linalg.svd(M)

                # # 防止数值超出 [-1, 1]
                # s = np.clip(s, -1.0, 1.0)
                # print(s)

                # # 计算主角
                # angles = np.arccos(s)


                # DISTRIBUTION[k][layer].append(np.max(angles))

                sim_matrix = np.abs(Vt1.T @ Vt2)

                # 每个方向取最大相似度
                max_sim_1 = np.max(sim_matrix, axis=1)  # U1中每个方向在U2中最相似的
                max_sim_2 = np.max(sim_matrix, axis=0)  # U2中每个方向在U1中最相似的

                angles = (np.mean(max_sim_1) + np.mean(max_sim_2)) / 2
                angles = np.degrees(np.arccos(np.clip(angles, 0, 1)))
                DISTRIBUTION[k][layer].append(angles)

            print(angles)
    # DISTRIBUTION = np.array(DISTRIBUTION)
    
    for layer in range(0,28,5):
        plt.figure(figsize=(20, 8))
        for k in range(100,1200,200):
            plt.plot([f'{STEPS[s-1]}/{STEPS[s]}' for s in range(1,len(STEPS))], DISTRIBUTION[k][layer], label = f'angle-of-final-{k}-vectors')
        plt.legend()
        plt.ylabel('ANGLES(°)')
        plt.xlabel('STEPS RELATION')
        plt.title('ANGLES CHANGES')
        plt.savefig(f'0623_pngs/show_angles_steps_{TASK}_{expert_part}_layer{layer}_expert{expert_id}.png')
        plt.close()



def show_cluster_steps1(STEPS,TASK):
    # RELATION = [[] for i in range(len(STEPS))]
    DISTRIBUTION = [[[] for l in range(28)] for k in range(1200)]
    output_dir = os.path.dirname('0626_pngs/show_distribution_steps.png')
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    expert_part = 'up'
    expert_id = 'all'
    # EVECTORS = []
    # for expert_id in range(16):

    for layer in range(28):
        for k in [1,2,5,10,20,50,100,500,1024]:
            EVECTORS = []
            for i in range(len(STEPS)):
                if STEPS[i] == 0 and TASK != 'CP':
                    parameter_matrix1 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_cp_oracle18000_CP_{expert_part}.tensors')
                else:
                    # parameter_matrix1 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_cp_oracle{STEPS[i-1]}_{TASK}_{expert_part}.tensors')
                    parameter_matrix1 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_sft_oracle{STEPS[i]}_{TASK}_{expert_part}.tensors')

                if len(parameter_matrix1)>1:
                    parameter_matrix01 = torch.cat(parameter_matrix1[layer],dim = 0).float().cpu().numpy()
                    # parameter_matrix01 = parameter_matrix1[layer][expert_id].float().cpu().numpy()

                else:
                    parameter_matrix01 = parameter_matrix1[layer][0].float().cpu().numpy()
                _, _, Vt1 = np.linalg.svd(parameter_matrix01, full_matrices=False)
                EVECTORS.append(Vt1[k-1])
            EVECTORS = np.array(EVECTORS)
            print(EVECTORS.shape)
            U = umap.UMAP(n_components=2).fit(EVECTORS)
            EVECTORS = U.transform(EVECTORS)
            plt.figure(figsize=(5, 5))
            for i in range(len(STEPS)):
                plt.scatter(EVECTORS[i,0],EVECTORS[i,1],c= 'b',alpha = (i+1)*0.1, label = f'ckpt-{STEPS[i]}')
            plt.ylabel('UMAP DIM 2')
            plt.xlabel('UMAP DIM 1')
            plt.legend()
            plt.title(f'UMAP {TASK} layer{layer}')
            plt.savefig(f'0626_pngs/show_cluster_steps_k{k}_{TASK}_{expert_part}_layer{layer}.png')
            plt.close()



def show_cluster_steps2(STEPS,TASK):
    # RELATION = [[] for i in range(len(STEPS))]
    DISTRIBUTION = [[[] for l in range(28)] for k in range(1200)]
    output_dir = os.path.dirname('0626_pngs/show_distribution_steps.png')
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    expert_part = 'up'
    expert_id = 'all'
    # EVECTORS = []
    # for expert_id in range(16):
    for k in [1,2,5,10,20,50,100,500,1024]:
        for i in range(len(STEPS)):
            EVECTORS = []
            
            if STEPS[i] == 0 and TASK != 'CP':
                parameter_matrix1 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_cp_oracle18000_CP_{expert_part}.tensors')
            else:
                parameter_matrix1 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_cp_oracle{STEPS[i-1]}_{TASK}_{expert_part}.tensors')
                # parameter_matrix1 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_sft_oracle{STEPS[i]}_{TASK}_{expert_part}.tensors')
            for layer in range(28):
                print(f'{k},{i},{layer}')
                if len(parameter_matrix1)>1:
                    parameter_matrix01 = torch.cat(parameter_matrix1[layer],dim = 0).float().cpu().numpy()
                    # parameter_matrix01 = parameter_matrix1[layer][expert_id].float().cpu().numpy()

                else:
                    parameter_matrix01 = parameter_matrix1[layer][0].float().cpu().numpy()
                _, _, Vt1 = np.linalg.svd(parameter_matrix01, full_matrices=False)
                EVECTORS.append(Vt1[k-1])
            EVECTORS = np.array(EVECTORS)
            print(EVECTORS.shape)
            U = umap.UMAP(n_components=2).fit(EVECTORS)
            EVECTORS = U.transform(EVECTORS)
            plt.figure(figsize=(10, 10))
            for l in range(28):
                plt.scatter(EVECTORS[l,0],EVECTORS[l,1],c= 'b',alpha = (l+1)*0.03, label = f'layer-{l}')
            plt.ylabel('UMAP DIM 2')
            plt.xlabel('UMAP DIM 1')
            plt.legend()
            plt.title(f'UMAP {TASK} STEPS{i}')
            plt.savefig(f'0626_pngs/show_cluster_layers_k{k}_{TASK}_{expert_part}_step{STEPS[i]}.png')
            plt.close()

KS = [1]
def show_layer_steps(STEPS,TASK):
    # RELATION = [[] for i in range(len(STEPS))]
    DISTRIBUTION = [[[] for l in range(28)] for k in range(1200)]
    output_dir = os.path.dirname('0626_pngs/show_distribution_steps.png')
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    expert_part = 'up'
    expert_id = 'all'
    # EVECTORS = []
    # for expert_id in range(16):
    for i in range(len(STEPS)):
        L = [[] for k in range(len(KS))]
        for k0 in range(len(KS)):
            # print(f'{i}, {k}')
            k = KS[k0]
            if STEPS[i] == 0 and TASK != 'CP':
                parameter_matrix1 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_cp_oracle18000_CP_{expert_part}.tensors')
            else:
                parameter_matrix1 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_cp_oracle{STEPS[i-1]}_{TASK}_{expert_part}.tensors')
                # parameter_matrix1 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0614/experts_sft_oracle{STEPS[i]}_{TASK}_{expert_part}.tensors')
            EVECTORS = []
            for layer in range(28):
                print(f'{k},{i},{layer}')
                if len(parameter_matrix1)>1:
                    parameter_matrix01 = torch.cat(parameter_matrix1[layer],dim = 0).float().cpu().numpy()
                    # parameter_matrix01 = parameter_matrix1[layer][expert_id].float().cpu().numpy()

                else:
                    parameter_matrix01 = parameter_matrix1[layer][0].float().cpu().numpy()
                _, _, Vt1 = np.linalg.svd(parameter_matrix01, full_matrices=False)
                EVECTORS.append(Vt1[k-1])
            EVECTORS = np.array(EVECTORS)
            for l in range(1,28):
                L[k0].append(cosine_similarity(EVECTORS[l-1].reshape(1, -1),EVECTORS[l].reshape(1, -1)).reshape(-1))
                # print(cosine_similarity(EVECTORS[i-1].reshape(-1, 1),EVECTORS[i].reshape(-1, 1)))
        plt.figure(figsize=(20, 20))
        c = 0
        for k in KS:
            plt.plot([f'{i}/{i+1}' for i in range(27)],L[c],c= 'b',alpha = (c+1)*0.1, label = f'first-{k}-sigvec')
            c+=1
        plt.ylabel('cosine similarity')
        plt.xlabel('layer pairs')
        plt.legend()
        plt.title(f'COSINESIMILARITY {TASK} STEPS{i}')
        plt.savefig(f'0626_pngs/show_layer_steps_k{k}_{TASK}_{expert_part}_step{STEPS[i]}.png')
        plt.close()


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






def get_small_data_ids(model,test_data):
    model.to(DEVICE)
    model.eval()
    dataloader = DataLoader(test_data, batch_size=1, num_workers=1, shuffle=False)
    # Q_PROJ = torch.load('/home/jxzhou/PLM_PER/qwen/0614/q_proj_qwen.tensors')[21][0].to(DEVICE).clone().detach()
    # u, s, V = torch.linalg.svd(Q_PROJ, full_matrices=False)
    # VS = V.T
    # TEM = 800
    # tensorA = torch.empty(1024, 2)
    # tensorA[:, 0] = float('inf')   # min
    # tensorA[:, 1] = float('-inf')  # max
    # tensorB = torch.zeros(TEM)

    # M1 = 700
    # M2 = 512
    # SIZE = 100

    # IDS = []
    for i, input_ids in enumerate(dataloader):
        # if i >SIZE:
        #     break
        # print(i)
        datas = input_ids.to(model.device)
        start_time = time.time()
        output = model(input_ids = datas, output_hidden_states=True, output_attentions=False, labels = datas,output_expert_labels =     False )
        stop_time = time.time()
        times = stop_time-start_time
        print(f"耗时: {times:.4f} 秒")
    #     # print(datas[0][-10:])
    #     hidden_states = output.hidden_states[21].clone().detach()##layerindex=20,q_proj_layerindex = 21

    #     PROJ = hidden_states @ VS
    #     PROJ = PROJ * s
    #     PROJ = PROJ.view(-1,1024)

    #     # matched_id, tensorA = process_batch_and_return_mean(PROJ, sample_id=i, tensorA=tensorA.to(PROJ.device), M=M1,K = M2)
    #     matched_id, tensorB = process_batch_and_return_mean2(PROJ[:,:TEM], sample_id=i, tensorB=tensorB.to(PROJ.device), M=M1,K = M2,rate=0.81)
        
    #     if matched_id:
            
    #         IDS.append(matched_id)
    #         print(matched_id,len(IDS))
    # torch.save(IDS,f'0717-deleted-data-ids-{int(len(IDS)/SIZE*100)}.pth')
    # with open(f"0717-deleted-data-ids-{int(len(IDS)/SIZE*100)}.txt", "w") as f:
    #     for row in IDS:
    #         f.write(str(row) + "\n")



        




        







if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)
    # # print(tokenizer.decode(16664))
    # qwen = prepare_qwenmodel()
    qwen = AutoModelForCausalLM.from_pretrained('/home/jxzhou/PLM_PER/qwen/Qwen3-0.6B', device_map="cuda:7", trust_remote_code=True)
    # # if TASK == 'MMLU' or TASK == 'BBH':
    # #     test_data = DATASET[TASK](f'/home/jxzhou/PLM_PER/datasets/{TASK}/{subset_dirs[0]}/test', tokenizer,n_shots=1)
    # # else:
    # #     test_data = DATASET[TASK](f'/home/jxzhou/PLM_PER/datasets', tokenizer,n_shots=1) ###ultrachat

    # # print(tokenizer.encode(':'))
    # print(qwen)
    # # analysis(test_data, qwen,MODEL_NAME,TASK)
    # get_experts(qwen,'qwen',TASK)
    test_data = Tokenized_data(tokenizer)
    get_small_data_ids(qwen,test_data)
    


    # # qwen_expert_matrix = torch.load('/home/jxzhou/PLM_PER/qwen/0612/experts_qwen.tensors')
    # # qwen_hiddenstates = torch.load('/home/jxzhou/PLM_PER/qwen/0612/hiddenstates_MMLU_qwen.tensors')
    # # qwen_inputs = torch.load('/home/jxzhou/PLM_PER/qwen/0612/inputs_MMLU_qwen.tensors')
    # # qwen_indices = torch.where(qwen_inputs[0] == 25)[0]


    # # oracle_expert_matrix = torch.load('/home/jxzhou/PLM_PER/qwen/0612/experts_oracle.tensors')
    # # oracle_hiddenstates = torch.load('/home/jxzhou/PLM_PER/qwen/0612/hiddenstates_MMLU_oracle.tensors')
    # # oracle_inputs = torch.load('/home/jxzhou/PLM_PER/qwen/0612/inputs_MMLU_oracle.tensors')
    # # oracle_indices = (oracle_inputs[0] == 25).nonzero(as_tuple=False).squeeze()


    # # switch_expert_matrix = torch.load('/home/jxzhou/PLM_PER/qwen/0612/experts_switch.tensors')
    # # switch_hiddenstates = torch.load('/home/jxzhou/PLM_PER/qwen/0612/hiddenstates_MMLU_switch.tensors')
    # # switch_inputs = torch.load('/home/jxzhou/PLM_PER/qwen/0612/inputs_MMLU_switch.tensors')
    # # switch_indices = (switch_inputs[0] == 25).nonzero(as_tuple=False).squeeze()


    # sft_oracle_expert_matrix = torch.load('/home/jxzhou/PLM_PER/qwen/0612/experts_sft_oracle.tensors')
    # sft_oracle_hiddenstates = torch.load('/home/jxzhou/PLM_PER/qwen/0612/hiddenstates_MMLU_sft_oracle.tensors')
    # sft_oracle_inputs = torch.load('/home/jxzhou/PLM_PER/qwen/0612/inputs_MMLU_sft_oracle.tensors')
    # sft_oracle_indices = (sft_oracle_inputs[0] == 25).nonzero(as_tuple=False).squeeze()


    # # sft_switch_expert_matrix = torch.load('/home/jxzhou/PLM_PER/qwen/0612/experts_sft_switch.tensors')
    # # sft_switch_hiddenstates = torch.load('/home/jxzhou/PLM_PER/qwen/0612/hiddenstates_MMLU_sft_switch.tensors')
    # # sft_switch_inputs = torch.load('/home/jxzhou/PLM_PER/qwen/0612/inputs_MMLU_sft_switch.tensors')
    # # sft_switch_indices = (sft_switch_inputs[0] == 25).nonzero(as_tuple=False).squeeze()






    # for layer in range(28):
    #     # visualize_projection_distribution(qwen_expert_matrix[layer],qwen_hiddenstates[layer],qwen_indices,20,'qwen',layer)
    #     # visualize_projection_distribution(oracle_expert_matrix[layer],oracle_hiddenstates[layer],oracle_indices,20,'oracle',layer)
    #     # # visualize_projection_distribution(switch_expert_matrix[layer],switch_hiddenstates[layer],switch_indices,20,'switch',layer)
    #     visualize_projection_distribution(sft_oracle_expert_matrix[layer],sft_oracle_hiddenstates[layer],sft_oracle_indices,20,'sft_oracle',layer)
    #     # visualize_projection_distribution(sft_switch_expert_matrix[layer],sft_switch_hiddenstates[layer],sft_switch_indices,20,'sft_switch',layer)



    #     # visualize_cluster_distribution(qwen_hiddenstates[layer], qwen_indices, 'qwen',layer)
    #     # visualize_cluster_distribution(oracle_hiddenstates[layer],oracle_indices,'oracle',layer)
    #     # visualize_cluster_distribution(switch_hiddenstates[layer],switch_indices,'switch',layer)
    #     # visualize_cluster_distribution(sft_oracle_hiddenstates[layer],sft_oracle_indices,'sft_oracle',layer)
    #     # visualize_cluster_distribution(sft_switch_hiddenstates[layer],sft_switch_indices,'sft_switch',layer)
    # 
    # STEPS = [0, 1000, 3000, 5000, 8000, 15000, 18000] ###CP
    
    # STEPS = [0, 100, 300, 500, 800, 1000, 3000] ###ULTRACHAT
    # STEPS = [10, 30, 50, 80] ###MMLU

    # show_distribution_relation_steps(STEPS,TASK)
    # show_distribution_steps(STEPS,TASK)
    # show_distribution_activate_expert_steps(STEPS,TASK)
    # show_distribution_count_steps(STEPS,TASK)
    # show_across_steps(STEPS,TASK)
    # show_cluster_steps2(STEPS,TASK)
    # show_cluster_steps1(STEPS,TASK)
    # show_layer_steps(STEPS,TASK)





