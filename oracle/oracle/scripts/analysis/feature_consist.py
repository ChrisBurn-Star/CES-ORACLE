from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
import torch
import os

from predictor import MSEPredictor, CEPredictor
from data_utils import Tokenized_data
from torch.optim import Adam, SGD

PREDICTOR_DEVICE = 'cuda:0'
NUM_EXPERTS = 60
HIDDEN_SIZE = 2048
LAYER_NUM = 24

section1 = [(0, 24)]
section2 = [(0, 12), (12, 24)]
section3 = [(0, 8), (8, 16), (16, 24)]
section4 = [(0, 6), (6, 12), (12, 18), (18, 24)]

section_2 = [(0,1), (1, 19), (19, 24)]

def cal_topk_acc(pred_topk, router_top4, is_reverse = False):
    # pred_logits: [batch_size * seq_len, topk]
    # router_logits: [batch_size * seq_len, 4]
    accs = []
    for pred, route in zip(pred_topk, router_top4):
        pred = set(pred.cpu().numpy())
        route = set(route.cpu().numpy())
        if is_reverse:
            all_experts = set(range(60))
            not_needed = all_experts - route
            not_pred = pred - route
            acc = len(not_needed & not_pred) / len(not_needed)
        else:
            acc = len(pred & route) / 4
            accs.append(acc)
    return sum(accs) / len(accs)


def draw_token_feature_consist(data, llm, tokenizer):
    token_embs = [[] for _ in range(LAYER_NUM)]
    for batch, text in enumerate(data, 1):
        qwen_inputs = tokenizer((text,), return_tensors="pt").to(qwen.device)
        with torch.no_grad():
            outputs = llm(input_ids = qwen_inputs.input_ids, output_router_logits=True, output_hidden_states=True)
        
        for i in range(LAYER_NUM):
            token_embs[i].append(outputs.hidden_states[i].reshape(-1, HIDDEN_SIZE)[:])
    
    us = []
    singulars = []
    vs = []
    singular_idx = 1
    # output = outputs.hidden_states[0].reshape(-1, HIDDEN_SIZE) # [batch, seq_len, vocab_size]
    output = torch.concat(token_embs[0], dim=0)
    feature_mat = output
    u, s0, v = torch.svd(feature_mat) # [N, N], [N], v.T = [N, D]
    s0 = s0 / (s0 **2).sum()
    singulars.append(s0.cpu().numpy())
    newu = s0[:singular_idx] @ u[:singular_idx]
    newu = newu / torch.norm(newu)
    newv = s0[:singular_idx] @ v.T[:singular_idx]
    newv = newv / torch.norm(newv)
    us.append(newu)
    vs.append(newv)

    for i in range(1, LAYER_NUM):
        print(f'layer {i}')
        # output = outputs.hidden_states[i].reshape(-1, HIDDEN_SIZE)
        output = torch.concat(token_embs[i], dim=0)
        feature_mat = output
        u, s0, v = torch.svd(feature_mat) # [N, N], [N], v.T = [N, D]
        s0 = s0 / (s0 **2).sum()
        singulars.append(s0.cpu().numpy())
        newu = s0[:singular_idx] @ u[:singular_idx]
        newu = newu / torch.norm(newu)
        newv = s0[:singular_idx] @ v.T[:singular_idx]
        newv = newv / torch.norm(newv)
        
        us.append(newu)
        vs.append(newv)
    
    us = torch.stack(us)
    vs = torch.stack(vs)
    cossim = (vs @ vs.T).abs()
    plt.imshow(cossim.cpu().numpy().T, cmap='hot', interpolation='nearest', vmin=0, vmax = 1)
    plt.colorbar()
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    plt.title(f'Feature Singular Consistency')
    plt.savefig(f'../figures/feature/token_feture_consist.png')
    plt.close()


def draw_sequence_feature_consist(data, llm, tokenizer):
    seq_embeddings = [[] for _ in range(LAYER_NUM)]
    for batch, text in enumerate(data, 1):
        qwen_inputs = tokenizer((text,), return_tensors="pt").to(qwen.device)
        with torch.no_grad():
            outputs = llm(input_ids = qwen_inputs.input_ids, output_router_logits=True, output_hidden_states=True)

        for i in range(LAYER_NUM):
            seq_embeddings[i].append(outputs.hidden_states[i].mean(dim=1, keepdim = False).reshape(HIDDEN_SIZE))


    us = []
    singulars = []
    vs = []
    singular_idx = 1
    
    output = torch.stack(seq_embeddings[0]) # [batch, seq_len, vocab_size]
    u, s0, v = torch.svd(output) # [N, N], [N], v.T = [N, D]
    s0 = s0 / (s0 **2).sum()
    singulars.append(s0.cpu().numpy())
    newu = s0[:singular_idx] @ u[:singular_idx]
    newu = newu / torch.norm(newu)
    newv = s0[:singular_idx] @ v.T[:singular_idx]
    newv = newv / torch.norm(newv)
    us.append(newu)
    vs.append(newv)

    for i in range(1, 24):
        print(f'layer {i}')
        output = torch.stack(seq_embeddings[i])
        u, s0, v = torch.svd(output) # [N, N], [N], v.T = [N, D]
        s0 = s0 / (s0 **2).sum()
        singulars.append(s0.cpu().numpy())
        newu = s0[:singular_idx] @ u[:singular_idx]
        newu = newu / torch.norm(newu)
        newv = s0[:singular_idx] @ v.T[:singular_idx]
        newv = newv / torch.norm(newv)
        
        us.append(newu)
        vs.append(newv)
    
    us = torch.stack(us)
    vs = torch.stack(vs)
    cossim = (vs @ vs.T).abs()
    plt.imshow(cossim.cpu().numpy().T, cmap='hot', interpolation='nearest', vmin=0, vmax = 1)
    plt.colorbar()
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    plt.title(f'Feature Singular Consistency')
    plt.savefig(f'../figures/feature/sequence_feture_consist.png')
    plt.close()



import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def draw_embedding(test_data, tokenizer, llm):
    text = test_data[1:3]
    with torch.no_grad():
        layer_hidden_states = [[] for _ in range(LAYER_NUM)]
        layer_router_logits = [[] for _ in range(LAYER_NUM)]
        for i in range(1, 3):
            text = test_data[i]
            qwen_inputs = tokenizer((text,), return_tensors="pt").to(llm.device)
            # output = llm(input_ids = qwen_inputs.input_ids, output_router_logits=True, output_attn_outputs=True) # 10/20：我改代码了，现在其实是输出  input hidden states
            output = llm(input_ids = qwen_inputs.input_ids, output_router_logits=True, output_hidden_states=True) # output hidden states 就是我要的每层输入
            for layer in range(LAYER_NUM):
                layer_hidden_states[layer].append(output.hidden_states[layer])
                layer_router_logits[layer].append(output.router_logits[layer])
            # output.router_logits # tuple([batch_size * seq_len, num_experts]), len = 24, num_experts = 60
            # output.attn_outputs # tuple([batch_size, seq_len, hidden_size]), len = 24

        for start_layer in range(LAYER_NUM):
            # print(f"Epoch {epoch}, Batch {batch}, Layer {start_layer}")
            # hidden_states = output.hidden_states[start_layer].reshape(-1, HIDDEN_SIZE).to(PREDICTOR_DEVICE)
            # router_logits = output.router_logits[start_layer].reshape(-1, NUM_EXPERTS).to(PREDICTOR_DEVICE)
            # breakpoint()
            hidden_states = torch.concat(layer_hidden_states[start_layer], 1).reshape(-1, HIDDEN_SIZE).to(PREDICTOR_DEVICE)
            # breakpoint()
            router_logits = torch.concat(layer_router_logits[start_layer], 0).reshape(-1, NUM_EXPERTS).to(PREDICTOR_DEVICE)
            expert_label = torch.argmax(router_logits, dim=1).cpu().numpy()
            # 计算每个 expert 在第一个 cluster 里选的多，还是第二个选的多
            choicea = []
            choiceb = []
            for expert_id in range(NUM_EXPERTS):
                a = (expert_label==expert_id)[:245].sum()
                b = (expert_label==expert_id)[245:].sum()
                choicea.append(a)
                choiceb.append(b)
            choicea = np.array(choicea)
            choiceb = -np.array(choiceb)

            plt.figure()
            plt.bar(range(NUM_EXPERTS), choicea, color='b', alpha=0.7)
            plt.bar(range(NUM_EXPERTS), choiceb, color='r', alpha=0.7)
            plt.savefig(f"../figures/expert_choice{start_layer}.png")
            plt.close()
            # hidden_states = hidden_states.cpu().numpy()
            # tsne = TSNE(n_components=2, random_state=0)
            # hidden_states = tsne.fit_transform(hidden_states)
            # plt.figure()
            # plt.scatter(hidden_states[:, 0], hidden_states[:, 1], c=expert_label, cmap='viridis', alpha=0.7)
            plt.savefig(f"../figures/{start_layer}.png")
            plt.close()
        breakpoint()
                # [batch_size * seq_len, hidden_size]
                # print(f"Epoch {epoch}, Batch {batch}, Layer {i}, Accuracy {accuracy:.4f}")



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("../Qwen1.5-MoE-A2.7B")
    qwen = AutoModelForCausalLM.from_pretrained('../Qwen1.5-MoE-A2.7B', device_map="auto")
    qwen_param_num=  sum(p.numel() for p in qwen.parameters())

    train_data = Tokenized_data('openweb', total = 10000)
    test_data = Tokenized_data('openweb', total = 50, is_test=True)

    draw_sequence_feature_consist(test_data, qwen, tokenizer)


