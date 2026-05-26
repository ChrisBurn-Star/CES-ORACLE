from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
import torch
import scipy
import time

from old_data_utils import Tokenized_data

PREDICTOR_DEVICE = 'cuda:0'
NUM_EXPERTS = 60
HIDDEN_SIZE = 2048
LAYER_NUM = 25

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


def train(epoch, train_data, test_data, llm, tokenizer, pred_model, section, loss_func, optimizer, lr_scheduler, ckpt_dir):
    for batch, text in enumerate(train_data, 1):
        optimizer.zero_grad()
        qwen_inputs = tokenizer((text,), return_tensors="pt").to(qwen.device)
        with torch.no_grad():
            output = llm(input_ids = qwen_inputs.input_ids, output_router_logits=True, output_attn_outputs=True) # 我改代码了，output_attn_outputs=True，返回每层的输入
            # output = llm(input_ids = qwen_inputs.input_ids, output_router_logits=True, output_hidden_states=True) # output hidden states 就是我要的每层输出
            # output.router_logits # tuple([batch_size * seq_len, num_experts]), len = 24, num_experts = 60
            # output.attn_outputs # tuple([batch_size, seq_len, hidden_size]), len = 24

        # pred_logits = []
        # for start_layer, end_layer in section:
        #     # start_layer, end_layer = LAYER_NUM * i // section, LAYER_NUM * (i + 1) // section
        #     pred_input = output.attn_outputs[start_layer].reshape(-1, HIDDEN_SIZE).to(PREDICTOR_DEVICE)
        #     pred_logits.append(pred_model(pred_input, start_layer, end_layer)) # [layers, batch_size * seq_len, num_experts]

        # pred_logits = torch.cat(pred_logits, dim=0)

        loss = 0
        for start_layer in range(LAYER_NUM):
            pred_logits = []
            pred_input = output.attn_outputs[start_layer].reshape(-1, HIDDEN_SIZE).to(PREDICTOR_DEVICE)
            # pred_input = output.hidden_states[start_layer].reshape(-1, HIDDEN_SIZE).to(PREDICTOR_DEVICE)
            pred_logits.append(pred_model(pred_input, start_layer)) # [layers, batch_size * seq_len, num_experts]

            pred_logits = torch.cat(pred_logits, dim=0)

            # pred_router_logits = pred_router_logits.reshape(-1 , NUM_EXPERTS)
            router_logits = torch.stack(output.router_logits, dim=0).to(PREDICTOR_DEVICE)
            loss += pred_model.__class__.cal_loss(loss_func, router_logits, pred_logits)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        print(f"Epoch {epoch}, Batch {batch}, Loss {loss.item()}")
        if (batch) % 200 == 0:
            torch.save(pred_model.state_dict(), f"{ckpt_dir}/{epoch}.{batch}.pth")
            test(test_data, tokenizer, llm, pred_model, section, epoch, batch, ckpt_dir)
            test(test_data, tokenizer, llm, pred_model, section, epoch, batch, ckpt_dir, topk=6)

    torch.save(pred_model.state_dict(), f"{ckpt_dir}/{epoch}.{batch}.pth")


def test(test_data, tokenizer, llm, pred_model, section, epoch, batch, ckpt_dir, topk = 4):
    layer_wise_test_acc = [
        [
            [] for _ in range(len(pred_model.predictors))
        ] for _ in range(LAYER_NUM)
        ] # 从第 i 层开始，预测第 j 层的结果
    
    pred_model.load_state_dict(torch.load(f"{ckpt_dir}/{epoch}.{batch}.pth", weights_only=True))
    for test_batch, text in enumerate(test_data):
        with torch.no_grad():
            qwen_inputs = tokenizer((text,), return_tensors="pt").to(llm.device)
            output = llm(input_ids = qwen_inputs.input_ids, output_router_logits=True, output_attn_outputs=True) # 10/20：我改代码了，现在其实是输出  input hidden states
            # output = llm(input_ids = qwen_inputs.input_ids, output_router_logits=True, output_hidden_states=True) # output hidden states 就是我要的每层输入
            # output.router_logits # tuple([batch_size * seq_len, num_experts]), len = 24, num_experts = 60
            # output.attn_outputs # tuple([batch_size, seq_len, hidden_size]), len = 24

            for start_layer in range(LAYER_NUM):
                # print(f"Epoch {epoch}, Batch {batch}, Layer {start_layer}")
                pred_logits = []
                # start_layer, end_layer = LAYER_NUM * i // section, LAYER_NUM * (i + 1) // section
                pred_input = output.attn_outputs[start_layer].reshape(-1, HIDDEN_SIZE).to(PREDICTOR_DEVICE)
                # pred_input = output.hidden_states[start_layer].reshape(-1, HIDDEN_SIZE).to(PREDICTOR_DEVICE)
                pred_logits.append(pred_model(pred_input, start_layer)) # [layers, batch_size * seq_len, num_experts]

                pred_logits = torch.concat(pred_logits, dim=0)

                # calculate accuracy for pred and router_logits
                for target_layer in range(LAYER_NUM):
                    layer_pred_index = pred_model.__class__.cal_pred_index(pred_logits[target_layer], topk) # [batch_size * seq_len, topk]
                    router_index = torch.topk(output.router_logits[target_layer], k = 4)[1].to(PREDICTOR_DEVICE) # [batch_size * seq_len, 4]
                    accuracy = cal_topk_acc(layer_pred_index, router_index)
                    layer_wise_test_acc[start_layer][target_layer].append(accuracy)
                # print(f"Epoch {epoch}, Batch {batch}, Layer {i}, Accuracy {accuracy:.4f}")

    for i in range(LAYER_NUM):
        for j in range(len(pred_model.predictors)):
            print(f"Epoch {epoch}, Batch {batch}, Layer-{i}-to-{j}-top{topk}-Accuracy {sum(layer_wise_test_acc[i][j]) / len(layer_wise_test_acc[i][j]):.4f}")
    # print(f"Epoch {epoch}, Batch {batch}, Average-top{topk}-Accuracy {sum([sum(layer_wise_test_acc[i]) / len(layer_wise_test_acc[i]) for i in range(LAYER_NUM)]) / LAYER_NUM:.4f}")
    #     print(f"Epoch {epoch}, Batch {batch}, Layer-{i}-top{topk}-Accuracy {sum(layer_wise_test_acc[i]) / len(layer_wise_test_acc[i]):.4f}")
    # print(f"Epoch {epoch}, Batch {batch}, Average-top{topk}-Accuracy {sum([sum(layer_wise_test_acc[i]) / len(layer_wise_test_acc[i]) for i in range(LAYER_NUM)]) / LAYER_NUM:.4f}")


import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def draw_embedding(test_data, tokenizer, llm):
    with torch.no_grad():
        layer_router_logits = [[] for _ in range(LAYER_NUM)]
        for i in range(1, 3):
            text = test_data[i]
            qwen_inputs = tokenizer((text,), return_tensors="pt").to(llm.device)
            output = llm(input_ids = qwen_inputs.input_ids, output_router_logits=True, output_hidden_states=True) # output hidden states 就是我要的每层输入
            for layer in range(LAYER_NUM):
                layer_router_logits[layer].append(output.router_logits[layer])

        for start_layer in range(LAYER_NUM):
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
        breakpoint()


def draw_token_expert(test_data, tokenizer, llm):
    plt.rcParams['font.size'] = 15
    plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
    with torch.no_grad():
        for i in range(len(test_data)):
            text = test_data[i]
            qwen_inputs = tokenizer((text,), return_tensors="pt").to(llm.device)
            output = llm(input_ids = qwen_inputs.input_ids, output_router_logits=True, output_hidden_states=True) # output hidden states 就是我要的每层输入
            for layer in range(LAYER_NUM):
                router_logits = output.router_logits[layer].reshape(-1, NUM_EXPERTS).to(PREDICTOR_DEVICE) # [batch_size * seq_len, num_experts]
                expert_label = torch.topk(router_logits, k=4, dim=1)[1].cpu().numpy() # 
                # 计算 expert activation 随 token 的变化

                plt.figure()
                for expert_id in range(4):
                    plt.scatter(range(len(expert_label)), expert_label[:,0], label=f"Expert-{expert_id}", s = 1.5, c = np.random.rand(3,))
                plt.xlabel('Token Index')
                plt.ylabel('Expert Index')
                plt.title(f"Layer-{layer} Expert Activation")
                plt.savefig(f"../figures/expert_choice/token_expert_activation_{layer}.png")
                plt.close()
            breakpoint()


def draw_data_embedding(test_data, tokenizer, llm):
    hidden_states = [[] for _ in range(LAYER_NUM)]
    with torch.no_grad():
        for i, input_ids in enumerate(test_data):
            input_ids = input_ids.to(llm.device)
            output = llm(input_ids = input_ids, output_hidden_states=True) # output hidden states 就是我要的每层输入
            for layer in range(LAYER_NUM):
                hidden_states[layer].append(output.hidden_states[layer]) # .reshape(-1, HIDDEN_SIZE)
                # hidden_states[layer].append(output.hidden_states[layer].mean(1)) # .reshape(-1, HIDDEN_SIZE)
            if i > 15:
                break
    # hidden_states = [torch.cat(hidden_states[layer], 0).cpu().float() for layer in range(LAYER_NUM)]
    # breakpoint()
    torch.save(hidden_states, '../figures/embeddings/qwen_token_embeddings.tensors')
    # breakpoint()


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
        u_basis.append(u[:1408])
        sing_values.append(s[:1408])
        v_basis.append(v[:, :1408])
        if i == 0:
            experts[i] = u[:1408] @ torch.diag(s) @ v.T

    torch.save(sing_values, f'../figures/embeddings/qwen_sing_values_layer_{layer}.tensors')

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

    np.save(f'../figures/embeddings/qwen_sing_values_kl_layer_{layer}.tensors', sv_kl)
    draw_heatmap(np.array(sv_kl), f'Singular Value KL-Divergence Layer {layer}', f'../figures/embeddings/qwen_sing_values_kl_layer_{layer}.png')
    # Expert 两两 左/右奇异向量 的子空间对齐度
    np.save(f'../figures/embeddings/qwen_left_subspace_alignment_layer_{layer}.tensors', left_subspace_alignment)
    draw_heatmap(np.array(left_subspace_alignment), f'Left Subspace Alignment Layer {layer}', f'../figures/embeddings/qwen_left_subspace_alignment_layer_{layer}.png')
    np.save(f'../figures/embeddings/qwen_right_subspace_alignment_layer_{layer}.tensors', right_subspace_alignment)
    draw_heatmap(np.array(right_subspace_alignment), f'Right Subspace Alignment Layer {layer}', f'../figures/embeddings/qwen_right_subspace_alignment_layer_{layer}.png')
    # 专家 flatten 后两两余弦相似度
    np.save(f'../figures/embeddings/qwen_flatten_similarity_layer_{layer}.tensors',flatten_similarity)
    draw_heatmap(np.array(flatten_similarity), f'Flatten Similarity Layer {layer}', f'../figures/embeddings/qwen_flatten_similarity_layer_{layer}.png')
    # 专家两两范数差异
    np.save(f'../figures/embeddings/qwen_norm_diff_layer_{layer}.tensors', norm_diff)
    draw_heatmap(np.array(norm_diff), f'Norm Difference Layer {layer}', f'../figures/embeddings/qwen_norm_diff_layer_{layer}.png')


def draw_heatmap(data, title, save_path):
    plt.imshow(data)
    plt.colorbar()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("../Qwen1.5-MoE-A2.7B")
    qwen = AutoModelForCausalLM.from_pretrained('../Qwen1.5-MoE-A2.7B', device_map="auto")
    qwen_param_num=  sum(p.numel() for p in qwen.parameters())

    # backbone = qwen.model
    # backbone.layers[0].mlp.shared_expert
    # backbone.layers[0].mlp.experts[0].gate_proj  # [gate_proj, up_proj, down_proj]

    for layer in range(24):
        shared_expert = qwen.model.layers[layer].mlp.shared_expert
        experts = qwen.model.layers[layer].mlp.experts
        experts.insert(0, shared_expert)
        experts = [expert.gate_proj.weight.float() for expert in experts] # [gate_proj, up_proj, down_proj]
        with torch.no_grad():
            ana_experts(experts, layer)

    # test_data = Tokenized_data(['openweb', 'legal', 'med'], tokenizer, total=50, is_test=True)
    # test_data = DataLoader(test_data, batch_size=10, shuffle=False)

    # draw_data_embedding(test_data, tokenizer, qwen)

    # section = 1
    # lr = 5e-5
    # predictor_class = CEPredictor

    # task_name = f'{predictor_class.__name__}_lr{lr}_section4'
    # ckpt_dir = f'../checkpoints/{task_name}'

    # print(f"Start training {task_name}")
    # model = predictor_class(layers=24, input_dim=HIDDEN_SIZE, num_experts=NUM_EXPERTS).to(PREDICTOR_DEVICE)
    # loss_func = predictor_class.get_loss_func()

    # optimizer = Adam(model.parameters(), lr=5e-5)
    # lr_scheduler = get_linear_schedule_with_warmup(optimizer, 500, 1e4)
    # torch.cuda.empty_cache()
    # ckpt_dir = f'../checkpoints/{predictor_class.__name__}_lr{lr}_all_to_all_attnoutput_without_residual'

    # if not os.path.exists(ckpt_dir):
    #     os.makedirs(ckpt_dir)

    # model = predictor_class(layers=24, input_dim=HIDDEN_SIZE, num_experts=NUM_EXPERTS).to(PREDICTOR_DEVICE)
    # loss_func = predictor_class.get_loss_func()
    # optimizer = Adam(model.parameters(), lr=5e-5)
    # lr_scheduler = get_linear_schedule_with_warmup(optimizer, 500, 1e4)

    # train(0, train_data, test_data[:50], qwen, tokenizer, model, None, loss_func, optimizer, lr_scheduler, ckpt_dir)
    # test(test_data[:50], tokenizer, qwen, model, sections, 0, 10000, ckpt_dir)
        


