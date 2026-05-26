
import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import PubMedForLM,LegalForLM,Wikipedia,MoMoE_longtailed,MoMoE_MIXED_LEGAL_REVIEW,MoMoE_LARGE_SPECIFIC_GENERAL,Wikitxt103ForLM_80W,MoMoE_FEWER_SPECIFIC_GENERAL,MoMoE_MIXED_LEGAL_PUBMED,MoMoE_MIXED_0128,MoMoE_MIXED_WIKI103_0124,MoMoE_MIXED,MoMoE_MIXED_WARMUP,MoMoE_WIKI103,MoMoE_WIKI103_WARMUP,MixedData_1211,MixedData_1211_0,RestaurantforLM_1103, MixedData, MixedData_stage1, ACLForLM,old_MixedData_after_stage1, Mixdata_1103, Wikitext,ACLForLM_1103,Mixdata_1115,Review_1103,Wikitxt103ForLM_1103,Wikitxt103ForLM_0102_warmup,Wikitxt103ForLM_0102_bert,MixedData_0110_0,MixedData_0110_1
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim
from sklearn.decomposition import PCA
import torch
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import util
from accelerate import DistributedDataParallelKwargs as DDPK
from accelerate import load_checkpoint_and_dispatch
import seaborn as sns
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import gc
import time
from Dataset_new import PubMedForLM,LegalForLM,Wikipedia,MoMoE_longtailed,MoMoE_MIXED_LEGAL_REVIEW,MoMoE_LARGE_SPECIFIC_GENERAL,Wikitxt103ForLM_80W,MoMoE_FEWER_SPECIFIC_GENERAL,MoMoE_MIXED_LEGAL_PUBMED,MoMoE_MIXED_0128,MoMoE_MIXED_WIKI103_0124,MoMoE_MIXED,MoMoE_MIXED_WARMUP,MoMoE_WIKI103,MoMoE_WIKI103_WARMUP,MixedData_1211,MixedData_1211_0,RestaurantforLM_1103, MixedData, MixedData_stage1, ACLForLM,old_MixedData_after_stage1, Mixdata_1103, Wikitext,ACLForLM_1103,Mixdata_1115,Review_1103,Wikitxt103ForLM_1103,Wikitxt103ForLM_0102_warmup,Wikitxt103ForLM_0102_bert,MixedData_0110_0,MixedData_0110_1

import matplotlib.pyplot as plt
import umap





DEVICE = 6
MODULE_NAMES = ['attn', 'ffn1', 'ffn2']


def cal_grad_sim(model, layer, module, dataloader, metric_fn):
    model.requires_grad_(False)
    if module == 'attn':
        model.decoders[layer].attn.key.weight.requires_grad_(True)
    elif module == 'ffn1':
        model.decoders[layer].feed_fwd.feed_fwd[0].weight.requires_grad_(True)
    elif module == 'ffn2':
        model.decoders[layer].feed_fwd.feed_fwd[2].weight.requires_grad_(True)

    common_grad, long_grad, longlong_grad = [], [], []
    for batch, (source, target, labels) in enumerate(dataloader):
        if batch > 0:
            break
        model.zero_grad()
        source = source.to(DEVICE)
        target = target.to(DEVICE)
        output = model(source)

        for i in range(output.size(0)):
            loss = metric_fn(output[i].view(-1, 50257), target[i].view(-1))
            loss.backward(retain_graph=True)
            if module == 'attn':
                grad = model.decoders[layer].attn.key.weight.grad.detach().clone()
            elif module == 'ffn1':
                grad = model.decoders[layer].feed_fwd.feed_fwd[0].weight.grad.detach().clone()
            elif module == 'ffn2':
                grad = model.decoders[layer].feed_fwd.feed_fwd[2].weight.grad.detach().clone()

            if labels[i] == 0:
                common_grad.append(grad)
            elif labels[i] == 1:
                long_grad.append(grad)
            else:
                longlong_grad.append(grad)
            model.zero_grad()
            torch.cuda.empty_cache()

    with torch.no_grad():
        common_grad, long_grad, longlong_grad = torch.stack(common_grad), torch.stack(long_grad), torch.stack(longlong_grad)
        common_grad, long_grad, longlong_grad = common_grad.reshape(common_grad.size(0), -1), long_grad.reshape(long_grad.size(0), -1), longlong_grad.reshape(longlong_grad.size(0), -1)
        batch_avg_grad = torch.cat([common_grad, long_grad, longlong_grad]).mean(dim=0).reshape(1, -1)

        common_sim = (torch.sum(batch_avg_grad * common_grad, axis = 1) / torch.sum(common_grad ** 2, axis=1)).mean().item()
        long_sim = (torch.sum(batch_avg_grad * long_grad, axis = 1) / torch.sum(long_grad ** 2, axis=1)).mean().item()
        longlong_sim = (torch.sum(batch_avg_grad * longlong_grad, axis = 1) / torch.sum(longlong_grad ** 2, axis=1)).mean().item()
    
    del common_grad, long_grad, longlong_grad, batch_avg_grad, source, target, output
    torch.cuda.empty_cache()
    gc.collect()

    return common_sim, long_sim, longlong_sim


def cal_ntk(model, layer, module, dataloader, normalized=True):
    model.requires_grad_(False)
    if module == 'attn':
        model.decoders[layer].attn.key.weight.requires_grad_(True)
    elif module == 'ffn1':
        model.decoders[layer].feed_fwd.feed_fwd[0].weight.requires_grad_(True)
    elif module == 'ffn2':
        model.decoders[layer].feed_fwd.feed_fwd[2].weight.requires_grad_(True)

    jacobian = []
    for batch, (source, target, labels) in enumerate(dataloader):
        if batch > 0:
            break
        model.zero_grad()
        source = source.to(DEVICE)
        target = target.to(DEVICE)
        output = model(source) # [batch, seq_len, vocab_size]

        for i in range(output.size(0)):
            output[i:i+1].mean().backward(retain_graph=True)
            if module == 'attn':
                grad = model.decoders[layer].attn.key.weight.grad.detach().clone()
            elif module == 'ffn1':
                grad = model.decoders[layer].feed_fwd.feed_fwd[0].weight.grad.detach().clone()
            elif module == 'ffn2':
                grad = model.decoders[layer].feed_fwd.feed_fwd[2].weight.grad.detach().clone()

            grad = grad.flatten()
            if normalized:
                grad = grad / torch.norm(grad)
            jacobian.append(grad)
            model.zero_grad()

    jacobian = torch.stack(jacobian)
    ntk = jacobian @ jacobian.T
    # lambda_, _ = torch.linalg.eig(ntk)
    # lambda_max = lambda_.real[0].item()

    return ntk.detach().cpu().numpy()


def draw_gradsim_figure(layer, module, common_consist, longlong_consist):
    plt.title(f'layer {layer} {module} grad similarity')
    plt.plot(common_consist, label='common')
    plt.plot(longlong_consist, label='longtail')
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('grad similarity')
    plt.savefig(f'../figure_data/all_layer_module_grad_sim/layer_{layer}_{module}_grad_similarity.png')
    plt.close()


def draw_ntk_figure(module, lambda_max, normed=True):
    plt.title(f'{module} lambda max')
    for i in range(lambda_max.shape[0]):    
        plt.plot(lambda_max[i], 'b-', label=f'layer {i}', alpha = (i + 1) / lambda_max.shape[0])
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('lambda max')
    plt.savefig(f'../figure_data/lambda_max/{module}_lambda_max_normed.png' if normed else f'../figure_data/lambda_max/{module}_lambda_max_unnormed.png')
    plt.close()


# def main():
#     dataloader = DataLoader(OpenwebTestData(256), batch_size=80)
#     metric_fn = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=50256)

#     # 有 12*3 种模块，每个模块有 4 种梯度，算 100 步。
#     common_consist = [{module:[] for module in MODULE_NAMES} for _ in range(12)]
#     long_consist = [{module:[] for module in MODULE_NAMES} for _ in range(12)]
#     longlong_consist = [{module:[] for module in MODULE_NAMES} for _ in range(12)]

#     for layer in range(12):
#         for module in MODULE_NAMES:
#             for test_batch in range(0, 150000 + 1, 1500):
#                 start_time = time.time()
#                 model = get_gpt(DEVICE, 0, test_batch,  50257, 12, 768, 12, 256, f'../checkpoint/3-mixed-768d/baseline/')
#                 common_sim, long_sim, longlong_sim = cal_grad_sim(model, layer, module, dataloader, metric_fn)
#                 common_consist[layer][module].append(common_sim)
#                 long_consist[layer][module].append(long_sim)
#                 longlong_consist[layer][module].append(longlong_sim)
#                 del model
#                 torch.cuda.empty_cache()
#                 gc.collect()
#                 print(f'layer {layer} {module} test_batch {test_batch} time {time.time() - start_time:.2f}s')

#             common_consist_lm = np.array(common_consist[layer][module])
#             long_consist_lm = np.array(long_consist[layer][module])
#             longlong_consist_lm = np.array(longlong_consist[layer][module])
#             draw_gradsim_figure(layer, module, common_consist_lm, longlong_consist_lm)
#             np.save(f'../figure_data/all_layer_module_grad_sim/layer_{layer}_{module}_grad_similarity.npy', [common_consist_lm, long_consist_lm, longlong_consist_lm])


# def main2():
#     dataloader = DataLoader(OpenwebTestData(256), batch_size=80, shuffle=False)
#     module_names = MODULE_NAMES[0:3]
#     layers= range(12)
#     normalized = False

#     # 有 12*3 种模块，每个模块有 4 种梯度，算 100 步。
#     lambda_max = {module:{layer:[] for layer in layers} for module in module_names}
#     for test_batch in range(0, 150000 + 1, 1500):
#         start_time = time.time()
#         model = get_gpt(DEVICE, 0, test_batch,  50257, 12, 768, 12, 256, f'../checkpoint/3-mixed-768d/baseline/')
#         print(f'test_batch {test_batch} load model time {time.time() - start_time:.2f}s')
#         for layer in layers:
#             for module in module_names:
#                 # lambda_max_lm = cal_ntk(model, layer, module, dataloader, normalized=normalized)
#                 ntk = cal_ntk(model, layer, module, dataloader, normalized=normalized)
#                 lambda_max[module][layer].append(ntk)
#                 # print(f'layer {layer} {module} test_batch {test_batch} lambda_max {lambda_max_lm:.2f}')
#                 print(f'layer {layer} {module} test_batch {test_batch}')
#         print(f'test_batch {test_batch} time {time.time() - start_time:.2f}s')


#     lambda_max = {module:np.array([np.array(lm) for layer, lm in lambda_max[module].items()]) for module in module_names}
#     for module in module_names:
#         save_name = f'../figure_data/lambda_max/{module}_ntk_normed.npy' if normalized else f'../figure_data/lambda_max/{module}_ntk_unnormed.npy'
#         np.save(save_name, lambda_max[module])
#         # draw_ntk_figure(module, lambda_max[module])


def draw_ef_figure(module, eff_rank):
    plt.title(f'{module} eff rank')
    for i in range(eff_rank.shape[1]):
        plt.plot(eff_rank[:, i], 'b-', label=f'layer {i}', alpha = (i + 1) / eff_rank.shape[1])
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('eff rank')
    plt.savefig(f'../figure_data/effective_rank/{module}_effective_rank.png')
    plt.close()

def compute_svd_torch(matrix):
    U, Sigma, Vt = torch.svd(matrix)
    return U, Sigma, Vt

# 取前3个最大奇异值对应的特征向量并构造新的矩阵
def get_top_k_singular_vectors_torch(U, Sigma, Vt, k=3):
    U_k = U[:, :k]
    Sigma_k = torch.diag(Sigma[:k])
    Vt_k = Vt[:, :k]
    return U_k @ Sigma_k @ Vt_k.T

# 计算两个矩阵的余弦相似度
def cosine_similarity_torch(matrix1, matrix2):
    dot_product = torch.dot(matrix1.flatten(), matrix2.flatten())
    norm_matrix1 = torch.norm(matrix1)
    norm_matrix2 = torch.norm(matrix2)
    return dot_product / (norm_matrix1 * norm_matrix2)

# def effective_rank(mat):
#     with torch.no_grad():
#         _, sigma, _ = torch.svd(mat)
#         return sigma.cpu().numpy()
#         sigma_norm1 = sigma.abs().sum()
#         pk = sigma / sigma_norm1
#         h = -sum(pk * torch.log(pk))
#         rank = torch.exp(h)
#     return rank.item()


# def cal_effrank(model: GPT, dataloader):
#     attn_rank, ffn_rank = np.zeros((12, 768)), np.zeros((12, 768))
#     for batch, (source, target, labels) in enumerate(dataloader):
#         if batch > 0:
#             break
#         with torch.no_grad():
#             source = source.to(DEVICE)
#             attn_output = model.get_attn_output(source, 0)
#             attn_rank[0] = effective_rank(attn_output.reshape(-1,768))
#             layer_output = model.decoders[0].get_ffn_output(attn_output)
#             ffn_rank[0] = effective_rank(layer_output.reshape(-1,768))
#             for i in range(1, 12):
#                 attn_output = model.decoders[i].get_attn_output(layer_output)
#                 attn_rank[i] = effective_rank(attn_output.reshape(-1,768))
#                 layer_output = model.decoders[i].get_ffn_output(attn_output)
#                 ffn_rank[i] = effective_rank(layer_output.reshape(-1,768))

#     return attn_rank, ffn_rank


# def main3():
#     for batch_size in [10, 100, 500, 1000, 2000]:
#         dataloader = DataLoader(OpenwebTestData(256), batch_size=batch_size, shuffle=True)
#         model = get_gpt(DEVICE, 0, 0,  50257, 12, 768, 12, 256, f'../checkpoint/3-mixed-768d/baseline/')
        
#         # 有 12 种模块，每个模块有 4 种梯度，算 100 步。
#         attn_ranks, ffn_ranks = [], []
#         for test_batch in range(0, 150000 + 1, 1500):
#             start_time = time.time()
#             if test_batch > 0:
#                 model.load_state_dict(torch.load(f'../checkpoint/3-mixed-768d/baseline/0_{test_batch}.pth'))
#             print(f'test_batch {test_batch} load model time {time.time() - start_time:.2f}s')
#             attn_rank, ffn_rank = cal_effrank(model, dataloader)
#             # print('\nattn', end='\t')
#             # for ar in attn_rank:
#             #     print(f'{ar:.2f}', end='\t')
#             # print('\nffn', end='\t')
#             # for fr in ffn_rank:
#             #     print(f'{fr:.2f}', end='\t')
                
#             attn_ranks.append(attn_rank)
#             ffn_ranks.append(ffn_rank)
#             print(f'test_batch {test_batch} time {time.time() - start_time:.2f}s')

#         attn_ranks, ffn_ranks = np.array(attn_ranks), np.array(ffn_ranks)
#         np.save(f'../figure_data/effective_rank/attn_sigma.npy', attn_ranks)
#         np.save(f'../figure_data/effective_rank/ffn_sigma.npy', ffn_ranks)
#         # draw_ef_figure(f'attn{batch_size}', attn_ranks)
#         # draw_ef_figure(f'ffn{batch_size}', ffn_ranks)
#         break



def cal_jacobian(model, layer, module, dataloader, normalized=False):
    tt =78
    model.requires_grad_(False)
    if module == 'attention':
        model.bert.encoders.layers[layer].attention.dense.weight.requires_grad_(True)
    elif module == 'dense_1':
        model.bert.encoders.layers[layer].ffn.dense_1.weight.requires_grad_(True)
    elif module == 'dense_2':
        model.bert.encoders.layers[layer].ffn.dense_2.weight.requires_grad_(True)

    jacobian = []
    for batch, data in enumerate(dataloader):
        if batch == tt: # 只跑一个 batch 采样
            model.zero_grad()
            # source = source.to(DEVICE)
            # target = target.to(DEVICE)
            _, output,_ = model(**data) # [batch, seq_len, vocab_size]

            for i in range(output.size(0)):
                output[i:i+1].mean().backward(retain_graph=True)
                if module == 'attention':
                    grad = model.bert.encoders.layers[layer].attention.dense.weight.grad.detach().clone()
                elif module == 'dense_1':
                    grad = model.bert.encoders.layers[layer].ffn.dense_1.weight.grad.detach().clone()
                elif module == 'dense_2':
                    grad = model.bert.encoders.layers[layer].ffn.dense_2.weight.grad.detach().clone()

                grad = grad.flatten()
                if normalized:
                    grad = grad / torch.norm(grad)
                jacobian.append(grad)
                model.zero_grad()
        elif batch>tt:
            break
        else:
            continue

    jacobian = torch.stack(jacobian)
    u, s, v = torch.svd(jacobian)
    # ntk = jacobian @ jacobian.T
    # lambda_, _ = torch.linalg.eig(ntk)
    # lambda_max = lambda_.real[0].item()

    return s.detach().cpu().numpy()
    # return jacobian




def cal_jacobian_FEATURE(model, layer, module, dataloader, normalized=False):
    tt =78
    model.requires_grad_(False)

    jacobian = []
    for batch, data in enumerate(dataloader):
        if batch == tt: # 只跑一个 batch 采样
            # model.zero_grad()
            # source = source.to(DEVICE)
            # target = target.to(DEVICE)
            _, _, output = model(**data) # [batch, seq_len, vocab_size]

            print(torch.cat(output).shape)
            jacobian.append(torch.cat(output).mean(1))

        elif batch>tt:
            break
        else:
            continue

    jacobian = torch.stack(jacobian)
    print(jacobian)
    u, s, v = torch.svd(jacobian)
    # ntk = jacobian @ jacobian.T
    # lambda_, _ = torch.linalg.eig(ntk)
    # lambda_max = lambda_.real[0].item()

    return s.detach().cpu().numpy()

def draw_singular_figure(module, sing_val):
    # plt.title(f'{module} singular value')
    # for i in range(sing_val.shape[0]):
    #     plt.plot(sing_val[i], 'b-', label=f'layer {i + 1}', alpha = (i + 1) / sing_val.shape[0])
    # plt.legend()
    # plt.xlabel('step')
    # plt.ylabel('singular value')
    # plt.savefig(f'/home/jxzhou/PLM_PER/1227/{module}_singular.png')
    # plt.close()
    fig, ax = plt.subplots(figsize=(5, 4))
    for layer in range(12):
        ax.plot(sing_val[layer], 'r-' if layer in [0,1,2,3,4,5] else 'b-', 
            label=f'Layer {layer + 1}', alpha = ((layer % 6) + 1) / 6)
        # ax.plot([eff_rank(sig) for sig in ffn_sigma[layer]], label=f'FFN {layer}')
    ax.legend(loc='upper left', ncol=1, fontsize=12, bbox_to_anchor=(1, 1))
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$\sigma_{max}$ / $\sigma_{1}$')
    plt.title(r'layer-wise')
    plt.ylim(0, 0.25)
    plt.xlim(0, 20)
    plt.axhline(y=1, color='black', linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 1])

    # plt.show()
    fig.savefig(f'/home/jxzhou/PLM_PER/1227/0927-{module}_singular_max.png')
def draw_singular_figure1(module, sing_val):
    # plt.title(f'{module} singular value')
    # for i in range(sing_val.shape[0]):
    #     plt.plot(sing_val[i], 'b-', label=f'layer {i + 1}', alpha = (i + 1) / sing_val.shape[0])
    # plt.legend()
    # plt.xlabel('step')
    # plt.ylabel('singular value')
    # plt.savefig(f'/home/jxzhou/PLM_PER/1227/{module}_singular.png')
    # plt.close()
    fig, ax = plt.subplots(figsize=(5, 4))
    for layer in range(12):
        ax.plot(sing_val[layer], 'r-' if layer in [0,1,2,3,4,5] else 'b-', 
            label=f'Layer {layer + 1}', alpha = ((layer % 6) + 1) / 6)
        # ax.plot([eff_rank(sig) for sig in ffn_sigma[layer]], label=f'FFN {layer}')
    ax.legend(loc='upper left', ncol=1, fontsize=12, bbox_to_anchor=(1, 1))
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$\sigma_{1}$ / $\sigma_{2}$')
    plt.title(r'layer-wise')
    plt.ylim(0, 8)
    plt.xlim(0, 20)
    plt.axhline(y=1, color='black', linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 1])

    # plt.show()
    fig.savefig(f'/home/jxzhou/PLM_PER/1227/0921-{module}_singular_1d2.png')

def draw_singular_figure2(module, sing_val):
    # plt.title(f'{module} singular value')
    # for i in range(sing_val.shape[0]):
    #     plt.plot(sing_val[i], 'b-', label=f'layer {i + 1}', alpha = (i + 1) / sing_val.shape[0])
    # plt.legend()
    # plt.xlabel('step')
    # plt.ylabel('singular value')
    # plt.savefig(f'/home/jxzhou/PLM_PER/1227/{module}_singular.png')
    # plt.close()
    fig, ax = plt.subplots()
    for layer in range(12):
        ax.plot(sing_val[layer], 'r-' if layer in [0,1,2,3,4,5] else 'b-', 
            label=f'Layer {layer + 1}', alpha = ((layer % 6) + 1) / 6)
        # ax.plot([eff_rank(sig) for sig in ffn_sigma[layer]], label=f'FFN {layer}')
    ax.legend(loc='upper left', ncol=1, fontsize=12, bbox_to_anchor=(1, 1))
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$\sigma_{1}$ / $\sigma_{-1}$')
    plt.title(r'layer-wise')
    plt.ylim(0, 1)
    plt.xlim(0, 30)
    plt.axhline(y=1, color='black', linestyle='--')
    plt.tight_layout(rect=[0, 0, 1, 1])

    # plt.show()
    fig.savefig(f'/home/jxzhou/PLM_PER/1227/0921-{module}_singular_1d-1.png')

def main4(jacobian_normed, singular_normed, config):
    layers = range(0, 12, 1)
    # dataloader = MoMoE_MIXED_LEGAL_PUBMED(config = config).train_loader

    modules = [ 'dense_2']
    sing_val = {module : [[] for _ in layers] for module in modules}
    sing_val_feature = {module : [[] for _ in layers] for module in modules}
    dataloader1 = MoMoE_longtailed(config = config,KKK = 100000).train_loader1
    dataloader2 = MoMoE_longtailed(config = config,KKK = 100000).train_loader2

    dataloader3 = MoMoE_longtailed(config = config,KKK = 100000).train_loader3

    model = base_models.BertForMLM(config)
    for test_batch in range(300, 2300 + 1, 100):
        print(f'test_batch {test_batch}')
        if test_batch > 0:
            model.load_state_dict(torch.load(f'/home/jxzhou/PLM_PER/MODELS_0918/STEPS_{test_batch}.pth').state_dict())
            for layer in layers:
                print(f'layer {layer}')
                for module in modules:
                    jacobian = []
                    for batch, data in enumerate(dataloader1):
                        if batch == 78: # 只跑一个 batch 采样
                            _, _, output = model(**data) # [batch, seq_len, vocab_size]
                            # jacobian.append(output[layer].mean(1))
                            sing_val_feature[module][layer].append(output[layer].mean(1))

                        elif batch > 78:
                            break
                        else:
                            continue
                    for batch, data in enumerate(dataloader2):
                        if batch == 78: # 只跑一个 batch 采样
                            _, _, output = model(**data) # [batch, seq_len, vocab_size]
                            # jacobian.append(output[layer].mean(1))
                            sing_val_feature[module][layer].append(output[layer].mean(1))

                        elif batch > 78:
                            break
                        else:
                            continue
                    for batch, data in enumerate(dataloader3):
                        if batch == 78: # 只跑一个 batch 采样
                            _, _, output = model(**data) # [batch, seq_len, vocab_size]
                            # jacobian.append(output[layer].mean(1))
                            sing_val_feature[module][layer].append(output[layer].mean(1))

                        elif batch > 78:
                            break
                        else:
                            continue
                    sing_val_feature[module][layer] = torch.cat(sing_val_feature[module][layer])
    for module in modules:
        sing_val[module] = torch.tensor(sing_val[module])
        print(sing_val[module].shape)
        torch.save(sing_val[module],f'/home/jxzhou/PLM_PER/{module}_sig_val_unnormed.npy')
    for module in modules:
        sing_val[module] = torch.load(f'/home/jxzhou/PLM_PER/{module}_sig_val_unnormed.npy')
    
    for module in modules:
        for layer in range(12):
            for step in range(len(sing_val[module][layer])):
                X = sing_val[module][layer][step].mean(1)
                X = umap.UMAP(random_state=42).fit_transform(X)
                
                plt.figure(layer*100+step*1)
                plt.scatter(X[0:50,0],X[0:50,1], c = 'g')
                plt.scatter(X[50:100,0],X[50:100,1], c = 'r')
                plt.scatter(X[100:150,0],X[100:150,1], c = 'b')

                plt.savefig(f'0930-layer{layer}-step{step}-cluster.png')
                

if __name__ == '__main__':
    config = BertConfig.from_json_file('config/bert.json')
    main4(False, True, config)
    

