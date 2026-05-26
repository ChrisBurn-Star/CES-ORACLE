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
from old_data_utils import Tokenized_data, Tokenized_data_chat, Tokenized_data_locality, OpenThoughtsDataset
from sklearn.decomposition import PCA
import umap
from sklearn.metrics.pairwise import cosine_similarity
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer



DEVICE = 'cuda:0'
NUM_EXPERTS = 64
HIDDEN_SIZE = 2048
LAYER_NUM = 27
MAT = 'up'
DIR = '../figures/deepseek'


def cluster_svd_embedding_space(rank, model, source, reflayer, clust_num, svd_dims):
    model.eval()
    model = model.to(rank)

    embeddings = []
    with torch.no_grad():
        batch_size = 100
        source = source.to(rank)
        for i in range(0, len(source), batch_size):
            embedding, attn_scores = model.get_attn_output(source[i:i+batch_size], reflayer, need_attn_score = True)
            ca = DBSCAN(eps=0.1, min_samples=5, metric='precomputed')
            embedding, attn_scores = embedding.cpu().numpy(), attn_scores.cpu().numpy()
            for j in range(len(embedding)):
                sg_labels = ca.fit_predict(attn_scores[j].mean(0))
                sg_centers, _ = cal_cluster_center(embedding[j], sg_labels)
                embeddings.append(np.stack(sg_centers, axis=0))

            # embeddings.append(embedding.mean(1, keepdim = False).to('cpu').numpy())

    embeddings = np.concatenate(embeddings, axis=0) # [N, embed_dim]
    u, s, v = np.linalg.svd(embeddings, full_matrices=False)
    print(f'singular values: {s}')
    # use the first 2 singular vectors to transform the embeddings
    # pick dims in svd_dims in v

    down_dim_mat = v[svd_dims].T
    embeddings = embeddings @ down_dim_mat

    clust_alg = KMeans(n_clusters=clust_num, random_state=0, n_init=10).fit(embeddings)
    # cluster_labels = clust_alg.labels_
    # cal number of each cluster
    # unique, counts = np.unique(cluster_labels, return_counts=True)
    # count_dict = dict(zip(unique, counts))
    # sort count_dict by count
    # count_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))
    # print(count_dict)

    centers, radius = cal_cluster_center(embeddings, clust_alg.labels_)

    return centers, radius, down_dim_mat

def cal_cluster_center(embeddings, labels):
    labelwise_embeddings = {}
    for label, emb in zip(labels, embeddings):
        # print(label)
        if not label in labelwise_embeddings:
            labelwise_embeddings[label] = []
        labelwise_embeddings[label].append(emb)

    centers = []
    for i in range(len(labelwise_embeddings)):
        labelwise_embeddings[i] = np.array(labelwise_embeddings[i])
        centers.append(labelwise_embeddings[i].mean(0))

    radius = []
    for i in range(len(labelwise_embeddings)):
        radius.append(np.linalg.norm(labelwise_embeddings[i] - centers[i], axis=1).mean())

    return centers, radius

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

def draw_data_attention(test_data, llm, types):
    hidden_states = [[] for _ in range(LAYER_NUM)]
    attention_scores = [[] for _ in range(LAYER_NUM)]
    losses = []
    routings = [[] for _ in range(LAYER_NUM)]

    input_idss = []
    # COMMON = test_data.common
    # print(COMMON)
    test_data.good = types
    
    with torch.no_grad():
        # old_gate = torch.load(f'ds_old_gate_0422.tensors')
        # new_gate = torch.load(f'ds_new_gate_0422.tensors')
        # P = torch.load(f'ds_P_0422.tensors')
        # W_frobenius = torch.norm(P, p='fro').to('cpu')  # 计算 W 的 Frobenius 范数

        # # 生成随机矩阵并匹配范数
        # random_matrix = torch.randn(64, 64)  # 标准正态分布随机矩阵
        # current_frobenius = torch.norm(random_matrix, p='fro')  # 当前范数

        # # 缩放矩阵，使其范数与 W 相同
        # P = random_matrix * (W_frobenius / current_frobenius)


        # print([old_gate == new_gate])
        # print(deepseek.model.layers[26].mlp.experts[0].up_proj.weight.data)
        
        # llm.model.layers[26].transform_Parameters(P, new_gate)
        # llm.model.layers[26].mlp.wo_res = 1
        # for e in range(64):
        #     print(llm.model.layers[26].mlp.experts[e].gate_proj.weight.device)
        #     k2  = llm.model.layers[26].mlp.experts[e].gate_proj.weight.data
            
        #     print([k1 == k2])
        # for r in range(20):
        #     test_data.ratio = r
        for i, input_ids in enumerate(test_data):
            # print(i)
            if i == 20:
                break
            input_ids = input_ids.to('cuda:7')
            
            # print(llm.device)

            output = llm(input_ids = input_ids, output_hidden_states=True, output_attentions=True, labels = input_ids) # output hidden states 就是我要的每层输入
                # print(output.loss)
                # losses.append(output.loss)
        # for i, input_ids in enumerate(test_data):
        #     # print(i)
        #     # if i == 1:
        #     #     test_data.good = types
        #     # else:
        #     #     test_data.good = 'good'
        #     input_ids = input_ids.to('cuda:7')
            
        #     # print(llm.device)

        #     output = llm(input_ids = input_ids, output_hidden_states=True, output_attentions=True, labels = input_ids) # output hidden states 就是我要的每层输入
        #     # print(output.loss)
        #     losses.append(output.loss)
            for layer in range(1,LAYER_NUM):
                hidden_states[layer].append(output.hidden_states[layer].reshape(-1, HIDDEN_SIZE)) # .reshape(-1, HIDDEN_SIZE)
                # hidden_states[layer].append(output.hidden_states[layer+1].view(-1, 2048)) # .reshape(-1, HIDDEN_SIZE)
                # attention_scores[layer].append(output.attentions[layer][0])
                # routings[layer].append(output.attentions[layer])
                # print(output.attentions[layer].shape)

                # print(output.hidden_states[layer].shape)
                # draw_heatmap(output.attentions[layer].mean(0).mean(0).float().cpu().numpy(),f'attentionscore_data{i}_layer{layer}')
                # print(output.attentions[layer].shape)
            # print(i)
            # input_idss.append(input_ids)

            # if i > 0:
            #     break
    # for l in range(len(hidden_states)):
        # hidden_states[l] = torch.cat(hidden_states[l],dim = 0)
    torch.save(hidden_states, f'/home/jxzhou/PLM_PER/qwen/0609/hiddenstates.tensors')
    # torch.save(input_idss, f'../rebuttal/ds_input_ids_locality.tensors')
    # torch.save(COMMON, f'../rebuttal/ds_input_ids_common_locality.tensors')

    # torch.save(routings, f'ds_routing_t.tensors')
    # torch.save(losses, f'/home/jxzhou/PLM_PER/qwen/0510/losses_{types}_ratio.tensors')
    # torch.save(attention_scores, f'/home/jxzhou/PLM_PER/qwen/0510/attentionscores_{types}.tensors')
    
    # torch.save(attention_scores, f'../rebuttal/qwen_attention_scores_small.tensors')


def show_activatation():
    routing = torch.load(f'ds_routing_t.tensors')[26]
    for r in range(20):
        plt.figure(0)
        plt.plot([i for i in range(len(routing[r]))], routing[r].float().cpu().numpy()[:,0])
        plt.xlabel('tokens')
        plt.ylabel('expert id')
        plt.title('activation')
        plt.savefig(f'ds_activation_t_data{r}.png')
        plt.close()




def cluster_relation_matrix(rel_matrix,r =0.1):
    # 构造邻接矩阵，排除自环边
    n = rel_matrix.size(0)
    device = rel_matrix.device
    mask = ~torch.eye(n, dtype=torch.bool, device=device)
    adj_matrix = ((rel_matrix > r) | (rel_matrix.transpose(0, 1) > r)) & mask

    # 初始化并查集结构
    parent = torch.arange(n, device=device)
    rank = torch.zeros(n, dtype=torch.int, device=device)

    # 定义路径压缩的find函数
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # 路径压缩
            u = parent[u]
        return u

    # 定义按秩合并的union函数
    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_u] = root_v
                if rank[root_u] == rank[root_v]:
                    rank[root_v] += 1

    # 遍历所有边进行合并
    edges = torch.argwhere(adj_matrix)
    for edge in edges:
        u, v = edge[0].item(), edge[1].item()
        union(u, v)

    # 确定每个节点的根节点并生成连续的标签
    labels = torch.tensor([find(i) for i in range(n)], device=device)
    _, labels = torch.unique(labels, return_inverse=True)
    return labels.view(-1)


def ana_emb_distance(file):
    hidden_states = torch.load(f'{DIR}/{file}.tensors', weights_only=True)
    # breakpoint()
    for layer in range(LAYER_NUM):
        # hidden_states[layer] = torch.cat(hidden_states[layer], dim = 0)
        for data in range(12):
            token_embed = hidden_states[layer][data][0]
            print(token_embed.shape)

            # dist = token_embed @ token_embed.T # pairwise_distances(hidden_states[layer])
            dist = pairwise_distances(token_embed)

            dist = dist.clip(0, dist.max()) #/ dist.max()
            # dist =  dist/dist.max() #/ dist.max()

            draw_heatmap(dist.float().cpu().numpy(), f'deepseek_{file}_token_distance_data{data}_layer{layer}')


def ana_emb_distance_compare(file1,file2):
    hidden_states1 = torch.load(f'{file1}.tensors')
    hidden_states2 = torch.load(f'{file2}.tensors')


    

    # breakpoint()
    for layer in range(26,27):
        token_embed1 = hidden_states1[layer][10000:]
        print(token_embed1.shape)
        dist1 = pairwise_distances(token_embed1)
        dist1 = dist1.clip(0, 180) #/ dist.max()
        print(dist1.max())
        # dist1 = dist1/dist1.max()
        draw_heatmap_0422(dist1.float().cpu().numpy(), f'{file1}_layer{layer}')
    
        token_embed2 = hidden_states2[layer][10000:]
        print(token_embed2.shape)
        dist2 = pairwise_distances(token_embed2)
        dist2 = dist2.clip(0, 180) #/ dist.max()
        # dist2 = dist2/dist2.max()

        print(hidden_states1[layer] == hidden_states2[layer].to(hidden_states1[layer].device))
        draw_heatmap_0422(dist2.float().cpu().numpy(), f'{file2}_layer{layer}')

        draw_heatmap_0422(np.abs(dist2.float().cpu().numpy() - dist1.float().cpu().numpy()), f'compare(tran-orig)_layer{layer}')


def get_stablerank_change(file):
    hidden_states = torch.load(f'{DIR}/{file}.tensors', weights_only=True)
    # breakpoint()
    
    for data in range(12):
        # hidden_states[layer] = torch.cat(hidden_states[layer], dim = 0)
        S = []
        for layer in range(LAYER_NUM):
            token_embed = hidden_states[layer][data][0].float()
            _, s, _ = torch.svd(token_embed)
            # print(s.min())

            # stable_rank = s.sum()/s.max()
            s = (s - s.min()) / (s.max() - s.min())
            # S.append(stable_rank.cpu().numpy())

            S.append((s > 0.1).sum().cpu().numpy())
            # plt.figure(0)
            # plt.plot([i for i in range(len(s))], s.cpu().numpy())
            # plt.xlabel('Dims')
            # plt.ylabel('E-Values')
            # plt.savefig(f'{DIR}/{file}-evalues-change-data{data}-layer{layer}.png')
            # plt.close()
        plt.figure(0)
        plt.plot([i for i in range(LAYER_NUM)], S)
        plt.xlabel('Layers')
        plt.ylabel('Stable Rank')
        plt.savefig(f'{DIR}/{file}-stablerank-change-data{data}.png')
        plt.close()


def get_stablerank_change_sentence():
    hidden_states = torch.load(f'{DIR}/ds_attention_without_res_token_large.tensors', weights_only=True)
    # breakpoint()
    SS = []
    for layer in range(LAYER_NUM):
        S = []
        hidden_states[layer] = torch.cat(hidden_states[layer], dim = 0)
        for l in range(1000,4000,500):
            

            token_embed = hidden_states[layer][:l].float()
            _, s, _ = torch.svd(token_embed)
            # print(s.min())

            stable_rank = s.sum()/s.max()
            # s = (s - s.min()) / (s.max() - s.min())
            S.append(stable_rank.cpu().numpy())
        SS.append(S)
        # S.append((s > 0.1).sum().cpu().numpy())
            # plt.figure(0)
            # plt.plot([i for i in range(len(s))], s.cpu().numpy())
            # plt.xlabel('Dims')
            # plt.ylabel('E-Values')
            # plt.savefig(f'{DIR}/{file}-evalues-change-data{data}-layer{layer}.png')
            # plt.close()
    plt.figure(0)
    colors = plt.cm.viridis(np.linspace(0, 1, len(SS)))
    for l in range(0,LAYER_NUM,2):
        plt.plot([i for i in range(1000,4000,500)], SS[l], label=f'layer_{l}',color = colors[l])
    plt.xlabel('Data Number')
    plt.ylabel('Stable Rank')
    plt.legend()
    plt.savefig(f'{DIR}/deepseek-stablerank-change-attentionwithoutres-token-layerall.png')
    plt.close()


def get_stablerank_change_params():
    
    # breakpoint()
    
    S = []
    for layer in range(LAYER_NUM):
        params = torch.load(f'{DIR}/ds_upexpert_params_layer_{layer+1}.tensors', weights_only=True)
        # hidden_states[layer] = torch.cat(hidden_states[layer], dim = 0)

        params = torch.cat(params,dim = 0).detach()
        print(params.shape)
        _, s, _ = torch.svd(params)
        # print(s.min())

        stable_rank = s.sum()/s.max()
        # s = (s - s.min()) / (s.max() - s.min())
        S.append(stable_rank.cpu().numpy())

        # S.append((s > 0.1).sum().cpu().numpy())
            # plt.figure(0)
            # plt.plot([i for i in range(len(s))], s.cpu().numpy())
            # plt.xlabel('Dims')
            # plt.ylabel('E-Values')
            # plt.savefig(f'{DIR}/{file}-evalues-change-data{data}-layer{layer}.png')
            # plt.close()
    plt.figure(0)
    plt.plot([i for i in range(LAYER_NUM)], S)
    plt.xlabel('Layers')
    plt.ylabel('Stable Rank')
    plt.savefig(f'{DIR}/deepseek-stablerank-change-params.png')
    plt.close()


def get_umap(file):

    hidden_states = torch.load(f'{DIR}/{file}.tensors')
    # A = torch.cat(hidden_states[0],dim = 0)
    # reducer = umap.UMAP()
    # reducer.fit(A.float().cpu().numpy())
    for layer in range(LAYER_NUM):
        reducer = umap.UMAP()

        # embedding = reducer.fit_transform(torch.cat(hidden_states[layer],dim = 0).float().cpu().numpy())
        embedding = reducer.fit_transform(hidden_states[layer][0].view(-1,2048).float().cpu().numpy())
        
        plt.figure(0)
        plt.scatter(embedding[:, 0], embedding[:, 1])
        plt.title('umap_cluster')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()
        # plt.xlim(11,15)
        # plt.ylim(18,20)
        plt.savefig(f'{DIR}/umap_token_deepseek_{file}_layer{layer}.png')
        plt.close()

def get_semantic_groups_change():
    hidden_states = torch.load(f'{DIR}/ds_sequence_embeddings.tensors',map_location='cuda:1')
    attention_scores = torch.load(f'{DIR}/ds_attention_scores.tensors',map_location='cuda:1')
    for data in range(12):
        num_sg_centers = []
        num_pre_token = []
        for layer in range(LAYER_NUM):
            sg_labels = cluster_relation_matrix(attention_scores[layer][data][0].mean(0).float(), r=0.1)
            sg_centers, _ = cal_cluster_center(hidden_states[layer][data].float().cpu().numpy(), sg_labels.tolist())
            print(len(sg_centers))
            num_sg_centers.append(len(sg_centers))
            num_pre_token.append(int(len(sg_labels)/len(sg_centers)))

        # plt.figure(0)
        # plt.plot([i for i in range(LAYER_NUM)],num_sg_centers )
        # plt.xlabel('Layers')
        # plt.ylabel('Num of SGs')
        # plt.savefig(f'{DIR}/num-sgs-change-data{data}.png')
        # plt.close()
        plt.figure(0)
        plt.plot([i for i in range(LAYER_NUM)],num_pre_token )
        plt.xlabel('Layers')
        plt.ylabel('Average of tokens num in SGs')
        plt.savefig(f'{DIR}/num-tokens-insgs-change-data{data}.png')
        plt.close()


def get_semantic_groups_compare():
    hidden_states1 = torch.load('/home/jxzhou/PLM_PER/qwen/figures/deepseek/ds_sequence_embeddings.tensors',map_location='cuda:1')
    attention_scores1 = torch.load('/home/jxzhou/PLM_PER/qwen/figures/deepseek/ds_attention_scores.tensors',map_location='cuda:1')
    hidden_states2 = torch.load('/home/jxzhou/PLM_PER/qwen/figures/deepseek/qw_sentence_embeddings.tensors',map_location='cuda:2')
    attention_scores2 = torch.load('/home/jxzhou/PLM_PER/qwen/figures/deepseek/qw_attention_scores.tensors',map_location='cuda:2')
    print(attention_scores1[0][0].shape, attention_scores2[0][0].shape)
    
    for data in range(5):
        SG1 = []
        SG2 = []
        for layer in range(LAYER_NUM):
            draw_heatmap(attention_scores1[layer][data][0].mean(0).float().cpu().numpy(),f'ds-attention-data{data}-layer{layer}',mina = 0, maxa = 0.05)
            sg_labels1 = cluster_relation_matrix(attention_scores1[layer][data][0].mean(0).float(), r=0.08)
            sg_centers1, _ = cal_cluster_center(hidden_states1[layer][data].float().cpu().numpy(), sg_labels1.tolist())
            # print(len(sg_centers1))
            SG1+=sg_centers1
            # print(len(SG1))
            draw_heatmap(attention_scores2[layer][data][0].mean(0).float().cpu().numpy(),f'qw-attention-data{data}-layer{layer}',mina = 0, maxa = 0.05)
            sg_labels2 = cluster_relation_matrix(attention_scores2[layer][data][0].mean(0).float(), r=0.08)
            sg_centers2, _ = cal_cluster_center(hidden_states2[layer][data].view(-1,2048).float().cpu().numpy(), sg_labels2.tolist())
            # print(len(sg_centers2))
            SG2+=sg_centers2

            similarity_matrix = cosine_similarity(SG1, SG2)
            draw_heatmap(similarity_matrix,f'sg-cos-similarity-compare-data{data}-layer{layer}',mina = 0, maxa = 0.3)
        SG1 = np.array(SG1)
        SG2 = np.array(SG2)
        # print(SG1.shape,SG2.shape)
        similarity_matrix = cosine_similarity(SG1, SG2)
        draw_heatmap(similarity_matrix,f'sg-cos-similarity-compare-data{data}-alllayer',mina = 0, maxa = 0.3)









        # plt.figure(0)
        # plt.plot([i for i in range(LAYER_NUM)],num_sg_centers )
        # plt.xlabel('Layers')
        # plt.ylabel('Num of SGs')
        # plt.savefig(f'{DIR}/num-sgs-change-data{data}.png')
        # plt.close()



def get_gate_transformer_Q(Transformer_Layer):
    hidden_states1 = torch.load('/home/jxzhou/PLM_PER/qwen/figures/deepseek/ds_sequence_embeddings.tensors',map_location='cuda:1')
    attention_scores1 = torch.load('/home/jxzhou/PLM_PER/qwen/figures/deepseek/ds_attention_scores.tensors',map_location='cuda:1')
    
    
    SG1 = []

    
    for data in range(5):

        for layer in range(LAYER_NUM):
            if layer == Transformer_Layer:
                draw_heatmap(attention_scores1[layer][data][0].mean(0).float().cpu().numpy(),f'ds-attention-data{data}-layer{layer}',mina = 0, maxa = 0.05)
                sg_labels1 = cluster_relation_matrix(attention_scores1[layer][data][0].mean(0).float(), r=0.08)
                sg_centers1, _ = cal_cluster_center(hidden_states1[layer][data].float().cpu().numpy(), sg_labels1.tolist())
                # print(len(sg_centers1))
                SG1+=sg_centers1
    SG1 = np.array(SG1)
    print(SG1.shape)









def get_semantic_groups_difflocat():
    hidden_states = torch.load(f'{DIR}/ds_attention_without_res',map_location='cuda:1')
    attention_scores = torch.load(f'{DIR}/ds_attention_scores.tensors',map_location='cuda:1')
    for data in range(12):
        num_sg_centers = []
        num_pre_token = []
        for layer in range(LAYER_NUM):
            sg_labels = cluster_relation_matrix(attention_scores[layer][data][0].mean(0).float(), r=0.1)
            sg_centers, _ = cal_cluster_center(hidden_states[layer][data].float().cpu().numpy(), sg_labels.tolist())
            print(len(sg_centers))
            num_sg_centers.append(len(sg_centers))
            num_pre_token.append(int(len(sg_labels)/len(sg_centers)))

        plt.figure(0)
        plt.plot([i for i in range(LAYER_NUM)],num_pre_token )
        plt.xlabel('Layers')
        plt.ylabel('Average of tokens num in SGs')
        plt.savefig(f'{DIR}/num-tokens-insgs-change-data{data}.png')
        plt.close()


def get_clusrer():
    hidden_states = torch.load(f'{DIR}/ds_sequence_embeddings.tensors',map_location='cuda:1')
    attention_scores = torch.load(f'{DIR}/ds_attention_scores.tensors',map_location='cuda:1')

    
    # ca = DBSCAN(eps=0.00001, min_samples=2, metric='precomputed')
    for layer in range(LAYER_NUM):
        embeddings = []
        # num_sg_centers = []
        # embedding, attn_scores = torch.cat(hidden_states[layer], dim = 0), torch.cat(attention_scores[layer], dim = 0)
        embedding, attn_scores = hidden_states[layer], attention_scores[layer]
        print(len(embedding))
        for j in range(len(embedding)):
            # print(attn_scores[j].min())
            # sg_labels = ca.fit_predict(attn_scores[j].mean(0).mean(0).float().cpu().numpy())
            sg_labels = cluster_relation_matrix(attn_scores[j][0].mean(0).float(), r=0.1)

            sg_centers, radii = cal_cluster_center(embedding[j].float().cpu().numpy(), sg_labels.tolist())
            # print(sg_labels.shape, embedding[j].float().cpu().numpy().shape, sg_centers[0].shape)
            print(len(sg_centers))

            # embeddings.append(np.stack(sg_centers, axis=0))
            # # print(embeddings[-1].shape)
            # centers = np.array(sg_centers)
            # radii = np.array(radii)
            # # 使用PCA降维到2维
            # pca = PCA(n_components=2)
            # centers_2d = pca.fit_transform(centers)

            # # 创建图形
            # fig, ax = plt.subplots()

            # # 绘制每个中心点和半径
            # for center, radius in zip(centers_2d, radii):
            #     # 绘制中心点（蓝色）
            #     ax.scatter(center[0], center[1], color='blue', s=10)
            #     # 绘制圆（蓝色边框，无填充）
            #     circle = plt.Circle(center, radius, color='blue', fill=False)
            #     ax.add_patch(circle)

            # # 调整坐标轴范围以显示所有圆
            # x_min, x_max = centers_2d[:, 0].min(), centers_2d[:, 0].max()
            # y_min, y_max = centers_2d[:, 1].min(), centers_2d[:, 1].max()
            # max_radius = radii.max()
            # ax.set_xlim(x_min - max_radius, x_max + max_radius)
            # ax.set_ylim(y_min - max_radius, y_max + max_radius)

            # # 设置等比例显示防止圆形变形
            # ax.set_aspect('equal')
            # plt.title("Centers and Radii Visualization")
            # plt.savefig(f'{DIR}/Centers-and-Radii-Visualization-data{j}-layer{layer}.png')
            # plt.close()


        # embeddings.append(embedding.mean(1, keepdim = False).to('cpu').numpy())

        # embeddings = np.concatenate(embeddings, axis=0) # [N, embed_dim]
        # print(embeddings.shape)
        # print(embeddings.shape)
        # u, s, v = np.linalg.svd(embeddings, full_matrices=False)
        # # print(f'singular values: {s}')
        # # use the first 2 singular vectors to transform the embeddings
        # # pick dims in svd_dims in v

        # down_dim_mat = v[range(0,10)].T
        # embeddings = embeddings @ down_dim_mat

        # clust_alg = KMeans(n_clusters=3, random_state=0, n_init=10).fit(embeddings)
        # cluster_labels = clust_alg.labels_
        # reducer = umap.UMAP()
        # embedding = reducer.fit(embeddings).transform(embeddings)
        # plt.figure(0)
        # for i in range(3):
        #     plt.scatter(embedding[cluster_labels == i, 0], embedding[cluster_labels == i, 1], label = f'domain{i}', alpha=0.1)
        # plt.title('umap_cluster')
        # plt.xlabel('UMAP 1')
        # plt.ylabel('UMAP 2')
        # plt.legend()
        # # plt.xlim(-25,25)
        # # plt.ylim(-25,25)
        # plt.savefig(f'{DIR}/umap_cluster_layer{layer}.png')
        # plt.close()
        # cal number of each cluster
        # unique, counts = np.unique(cluster_labels, return_counts=True)
        # count_dict = dict(zip(unique, counts))
        # sort count_dict by count
        # count_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))
        # print(count_dict)

        # centers, radii = cal_cluster_center(embeddings, clust_alg.labels_)
        # # print(centers,radii)
        # centers = np.array(centers)
        # radii = np.array(radii)
        # # 使用PCA降维到2维
        # pca = PCA(n_components=2)
        # centers_2d = pca.fit_transform(centers)

        # # 创建图形
        # fig, ax = plt.subplots()

        # # 绘制每个中心点和半径
        # for center, radius in zip(centers_2d, radii):
        #     # 绘制中心点（蓝色）
        #     ax.scatter(center[0], center[1], color='blue', s=10)
        #     # 绘制圆（蓝色边框，无填充）
        #     circle = plt.Circle(center, radius, color='blue', fill=False)
        #     ax.add_patch(circle)

        # # 调整坐标轴范围以显示所有圆
        # x_min, x_max = centers_2d[:, 0].min(), centers_2d[:, 0].max()
        # y_min, y_max = centers_2d[:, 1].min(), centers_2d[:, 1].max()
        # max_radius = radii.max()
        # ax.set_xlim(x_min - max_radius, x_max + max_radius)
        # ax.set_ylim(y_min - max_radius, y_max + max_radius)

        # # 设置等比例显示防止圆形变形
        # ax.set_aspect('equal')
        # plt.title("Centers and Radii Visualization")
        # plt.savefig(f'{DIR}/Centers-and-Radii-Visualization-layer{layer}.png')
        # plt.close()
        # return centers, radius, down_dim_mat




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
    torch.save(experts,f'{DIR}/ds_{MAT}expert_params_layer_{layer}.tensors')
    # # Expert 两两 singular value 的分布 KL divergence
    # sv_kl = [[0 for _ in range(NUM_EXPERTS + 1)] for _ in range(NUM_EXPERTS + 1)]
    # left_subspace_alignment = [[0 for _ in range(NUM_EXPERTS + 1)] for _ in range(NUM_EXPERTS + 1)]
    # right_subspace_alignment = [[0 for _ in range(NUM_EXPERTS + 1)] for _ in range(NUM_EXPERTS + 1)]
    # flatten_similarity = [[0 for _ in range(NUM_EXPERTS + 1)] for _ in range(NUM_EXPERTS + 1)]
    # norm_diff = [[0 for _ in range(NUM_EXPERTS + 1)] for _ in range(NUM_EXPERTS + 1)]
    # for i in range(NUM_EXPERTS + 1):
    #     t = time.time()
    #     for j in range(i, NUM_EXPERTS + 1):
    #         sv_kl[i][j] = torch.nn.functional.kl_div(sing_values[i].log(), sing_values[j], reduction='sum').item()
    #         left_subspace_alignment[i][j] = subspace_angles_torch(u_basis[i], u_basis[j]).item()
    #         right_subspace_alignment[i][j] = subspace_angles_torch(v_basis[i], v_basis[j]).item()
    #         flatten_similarity[i][j] = torch.nn.functional.cosine_similarity(experts[i].flatten(), experts[j].flatten(), dim=0).item()
    #         norm_diff[i][j] = (torch.norm(experts[i], p=2) - torch.norm(experts[j], p=2)).item()
    #     print(f"Time for expert {i}: {time.time() - t}")

    # np.save(f'{DIR}/ds_{MAT}sing_values_kl_layer_{layer}.tensors', sv_kl)
    # draw_heatmap(np.array(sv_kl), f'Singular Value KL-Divergence Layer {layer}', f'{DIR}/ds_{MAT}sing_values_kl_layer_{layer}.png')
    # # Expert 两两 左/右奇异向量 的子空间对齐度
    # np.save(f'{DIR}/ds_{MAT}left_subspace_alignment_layer_{layer}.tensors', left_subspace_alignment)
    # draw_heatmap(np.array(left_subspace_alignment), f'Left Subspace Alignment Layer {layer}', f'{DIR}/ds_{MAT}left_subspace_alignment_layer_{layer}.png')
    # np.save(f'{DIR}/ds_{MAT}right_subspace_alignment_layer_{layer}.tensors', right_subspace_alignment)
    # draw_heatmap(np.array(right_subspace_alignment), f'Right Subspace Alignment Layer {layer}', f'{DIR}/ds_{MAT}right_subspace_alignment_layer_{layer}.png')
    # # 专家 flatten 后两两余弦相似度
    # np.save(f'{DIR}/ds_{MAT}flatten_similarity_layer_{layer}.tensors',flatten_similarity)
    # draw_heatmap(np.array(flatten_similarity), f'Flatten Similarity Layer {layer}', f'{DIR}/ds_{MAT}flatten_similarity_layer_{layer}.png')
    # # 专家两两范数差异
    # np.save(f'{DIR}/ds_{MAT}norm_diff_layer_{layer}.tensors', norm_diff)
    # draw_heatmap(np.array(norm_diff), f'Norm Difference Layer {layer}', f'{DIR}/ds_{MAT}norm_diff_layer_{layer}.png')


def draw_heatmap(data, title, mina = 0, maxa = 1):
    plt.close()
    plt.imshow(data, cmap = 'hot', interpolation = 'nearest', vmin=mina, vmax= maxa)
    plt.colorbar()
    plt.title(title)
    plt.savefig(f'{DIR}/{title}.png')

def draw_heatmap_0422(data, title):
    plt.close()
    plt.imshow(data, cmap = 'hot', interpolation = 'nearest')
    plt.colorbar()
    plt.title(title)
    plt.savefig(f'{title}.png')


def pairwise_distances(x):
    # 展开x以准备进行广播相加
    xx = (x ** 2).sum(dim=1).view(-1, 1)
    xy = torch.mm(x, x.t())    # 计算点积并展开    
    distance_matrix = xx - 2 * xy + xx.t()    # 计算距离矩阵    
    # 因为可能有浮点数精度问题导致负数出现，所以这里需要将负数变成0
    distance_matrix = torch.clamp(distance_matrix, min=0)
    # 开方得到真实欧氏距离
    return torch.sqrt(distance_matrix)


def draw_icml_fig3(file,t):
    for layer in range(24):
        print(layer)
        for d in range(5,10):
            data = torch.load(f'../rebuttal/{file}.tensors')[layer][d].float().mean(0).mean(0)[:t,:t].cpu().numpy()
            plt.figure(0)
            plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
            plt.rcParams['font.size'] = 15
            plt.imshow(data[1:, 1:], cmap = 'hot', interpolation = 'nearest', vmin=0, vmax= 0.01)
            # plt.colorbar()
            plt.title('Attention Score')
            plt.xlabel('Token Index')
            plt.ylabel('Token Index')
            plt.savefig(f'../rebpngs/fig3_{file}_data{d}_layer{layer}.png')
            plt.close()


def draw_attention(file,types,highlight_indices):
    for layer in range(1,26,8):
        # print(layer)
        for d in range(1):
            data = torch.load(f'/home/jxzhou/PLM_PER/qwen/0510/{file}_{types}.tensors')[layer][d].mean(0).float().cpu().numpy()
            # print(data.shape)
            plt.figure(0)
            plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
            plt.rcParams['font.size'] = 15
            # plt.imshow(data[:, :], cmap = 'hot', interpolation = 'nearest')
            fig, ax = plt.subplots()
            im = ax.imshow(data[1:, 1:], cmap = 'hot', interpolation = 'nearest', vmin=0,vmax =0.2 )

            # 构造横纵坐标刻度
            xticks = list(range(data.shape[1]))
            yticks = list(range(data.shape[0]))

            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

            # 设置坐标标签
            ax.set_xticklabels(xticks)
            ax.set_yticklabels(yticks)

            # 获取当前x/y标签对象
            xtick_labels = ax.get_xticklabels()
            ytick_labels = ax.get_yticklabels()

            # 标红加粗指定的标签
            if highlight_indices:
                for i in highlight_indices[d]:
                    xtick_labels[i].set_color('red')
                    xtick_labels[i].set_fontweight('bold')
                    ytick_labels[i].set_color('red')
                    ytick_labels[i].set_fontweight('bold')
            # plt.colorbar()
            plt.title('Attention Score')
            plt.xlabel('Token Index')
            plt.ylabel('Token Index')
            plt.savefig(f'/home/jxzhou/PLM_PER/qwen/0510/attentionscores_data{d}_layer{layer}_{types}.png')
            plt.close()

def draw_icml_fig4(file):
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'pink', 'brown', 'gray']
    for layer in range(24):
        sgs = []
        sentence_tokens = torch.load(f'../rebuttal/{file}.tensors')[layer]
        print(len(sentence_tokens))
        for d in range(0,100,5):
            if d//5 >= len(colors):
                break
            tokens = sentence_tokens[d]
            for j in range(10, 200 + 1, 10):
                sgs.append(tokens[0][:j].mean(0).float().cpu().numpy())
        sgs = umap.UMAP(n_components=2).fit_transform(sgs)
        plt.figure(0)
        plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
        plt.rcParams['font.size'] = 15
        for i in range(0, len(sgs), 20):
            for j in range(0, 20):
                plt.scatter(sgs[i + j, 0], sgs[i + j, 1], color=colors[i // 20], s=10, alpha= (j + 1) / 20)
                
                

        plt.title('Semantic Group Embedding')
        # plt.xlim(-10,20)
        # plt.ylim(-10,20)
        plt.savefig(f'../rebpngs/fig4_{file}_layer{layer}.png')
        plt.close()


def draw_icml_cluster_compare(file1, file2):
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'pink', 'brown', 'gray']
    data1 = torch.load(f'../rebuttal/{file1}.tensors')
    data2 = torch.load(f'../rebuttal/{file2}.tensors')

    
    for layer in range(1,24):
        token_level_emneddings1 = []
        token_level_emneddings2 = []
        for d in range(0,100,20):
            # print(data1[layer-1][d].shape)

            token_level_emneddings1.append(data1[layer-1][d][:, 0:50,:].view(-1,2048))
            token_level_emneddings2.append(data2[layer][d][:, 0:50,:].view(-1,2048))
        
        token_level_emneddings1 = torch.cat(token_level_emneddings1, dim = 0).float().cpu().numpy()
        token_level_emneddings2 = torch.cat(token_level_emneddings2, dim = 0).float().cpu().numpy()
        print(token_level_emneddings1.shape)

        # U = umap.UMAP(n_components=2).fit(token_level_emneddings2)
        token_level_emneddings1 = umap.UMAP(n_components=2).fit_transform(token_level_emneddings1)
        token_level_emneddings2 = umap.UMAP(n_components=2).fit_transform(token_level_emneddings2)
        plt.figure(0)
        plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
        plt.rcParams['font.size'] = 15
        for data in range(10):
            plt.scatter(token_level_emneddings1[data*100:(data+1)*100, 0],token_level_emneddings1[data*100:(data+1)*100, 1], color = colors[data], alpha = 0.1)
        plt.title('Token-Level Cluster')
        plt.savefig(f'../rebpngs/fig5_{file1}_layer{layer}.png')
        plt.close()
        plt.figure(0)
        plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
        plt.rcParams['font.size'] = 15
        for data in range(10):
            plt.scatter(token_level_emneddings2[data*100:(data+1)*100, 0],token_level_emneddings2[data*100:(data+1)*100, 1], color = colors[data], alpha = 0.1)
        plt.title('Token-Level Cluster')
        plt.savefig(f'../rebpngs/fig5_{file2}_layer{layer}.png')
        plt.close()




def draw_icml_fig2(file, input_ids_file,tokenizer):
    T = 14
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'pink', 'brown', 'gray']
    datas = torch.load(f'../rebuttal/{file}.tensors')
    input_ids = torch.load(f'../rebuttal/{input_ids_file}.tensors')
    labels = []
    common_labels = [' study',  ' patterns' ,' systems', ' that', ' process', ' data', ' using', ' to', ' and', ' using']
    
    ZOOM_data_id = []
    ## study data patterns
    
    
    for i in range(len(input_ids)):
        # print(tokenizer.decode(input_ids[i][0]))
        item = []
        for j in range(len(input_ids[i][0])):
            item += [tokenizer.decode(input_ids[i][0][j])]
        labels.append(item)
    print(labels)
        # print(input_ids[i].shape)
        # labels.append(input_ids[i][0].float().cpu().numpy())
    for key in [' that', ' is', ' to', ' in', ' normally']:
        for layer in range(10):
            TEMP = []
            token_level_emneddings = []
            for d in range(40):
                # print(data1[layer-1][d].shape)

                token_level_emneddings.append(datas[layer][d][:, :,:].view(-1,2048))
            
            token_level_emneddings = torch.cat(token_level_emneddings, dim = 0).float().cpu().numpy()

            print(token_level_emneddings.shape)

            token_level_emneddings = umap.UMAP(n_components=2).fit_transform(token_level_emneddings)
            plt.figure(0)
            plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
            plt.rcParams['font.size'] = 15
            c = -1
            for data in range(40):
                # plt.scatter(token_level_emneddings[data*T:(data+1)*T, 0], token_level_emneddings[data*T:(data+1)*T, 1], color = colors[data], alpha = 0.1)
                for k in range(len(datas[layer][data][0])):
                    c+=1
                    # print(token_level_emneddings.shape)
                    
                    # if labels[data][k] in common_labels and labels[data][k] not in TEMP:
                    # if labels[data][k] in [' that', ' is', ' to', ' in', ' normally']:
                    if labels[data][k] == key:

                        plt.scatter(token_level_emneddings[c, 0], token_level_emneddings[c, 1], color = colors[data//10], alpha = 0.8)

                        # print(labels[data][k])
                        # TEMP.append(labels[data][k])
                        
                        # plt.annotate(labels[data][k], (token_level_emneddings[data*T+k, 0], token_level_emneddings[data*T+k, 1]), textcoords="offset points", xytext=(0,10), ha='center', alpha = 0.5)
                    # if labels[data][k] in [' that', ' is', ' to', ' in', ' normally'] and labels[data][k] not in TEMP:
                    #     TEMP.append(labels[data][k])
                    #     plt.annotate(labels[data][k], (token_level_emneddings[c, 0], token_level_emneddings[c, 1]), textcoords="offset points", xytext=(0,10), ha='center', alpha = 0.5)
            
            print(c)
            plt.title('Token-Level Cluster')
            plt.savefig(f'../rebpngs/fig2_key{key}_{file}_layer{layer}.png')
            plt.close()



def show_context_preference(domain):
    data = torch.load(f'../figures/deepseek/ds_attention_selfout_{domain}.tensors')
    routes = torch.load(f'../figures/deepseek/ds_attention_routing_{domain}.tensors')
    for l in range(10,11):
        EXPERTS = torch.load(f'../figures/deepseek/ds_upexpert_params_layer_{l}.tensors')
        context = data[l][20][0]
        US = []
        route = routes[l][20]
        print(context.shape)
        print(route.shape)
        UE = []

        for e in range(65):
            if e in route[5:8] and e not in UE:
                # print(e)
                _, _, V= torch.svd(EXPERTS[e])
                V = V.T[:5]
                US.append(V)
                UE.append(e)
        US = torch.cat(US)
        print(US.shape)

        plt.figure(0)
        for t in range(5,8):
            projection = torch.matmul(context[t].float(), US.T.float().to(context[t].device)).detach().cpu().numpy()
            # print(projection.shape)
            # plt.figure(0)
            plt.bar([b for b in range(projection.shape[0])], np.abs(projection), label = f'token{t}' , alpha = 0.2)
        plt.xlabel('vectors(in groups of five)')
        plt.ylabel('projection')
        plt.savefig(f'../0407/{domain}_projection_layer{l}_attention_out.png')
        plt.close()

def show_token_identity_preference():
    data1 = torch.load(f"../figures/deepseek/ds_attention_out_{'legal'}.tensors")
    routes1 = torch.load(f"../figures/deepseek/ds_attention_routing_out_{'legal'}.tensors")
    data2 = torch.load(f"../figures/deepseek/ds_attention_out_{'med'}.tensors")
    routes2 = torch.load(f"../figures/deepseek/ds_attention_routing_out_{'med'}.tensors")
    for l in range(10,11):
        EXPERTS = torch.load(f'../figures/deepseek/ds_upexpert_params_layer_{l}.tensors')
        # print(data1[l][20][0].shape)
        tokens1 = torch.stack([data1[l][20][0][14,:], data1[l][20][0][103,:]])

        # tensor([[  0,  30],
        # [  0,  49],
        # [  0,  57],
        # [  0,  67],
        # [  0, 133],
        # [  0, 161],
        # [  0, 196],
        # [  0, 211],
        # [  0, 254],
        # [  0, 260],
        # [  0, 284],
        # [  0, 297],
        # [  0, 327]])
        tokens2 = torch.stack([data2[l][20][0][138,: ], data2[l][20][0][218,:]])
        US = []
        route1 = routes1[l][20]
        route2 = routes2[l][20]

        # print(context.shape)
        print(tokens1.shape)
        UE = []

        for e in range(65):
            if e in route2[138] or e in route2[218] or e in route1[14] or e in route1[103] and e not in UE:
                # print(e)
                _, _, V= torch.svd(EXPERTS[e])
                V = V.T[:5]
                US.append(V)
                UE.append(e)
        US = torch.cat(US)
        print(US.shape)

        plt.figure(0)
        for t in range(2):
            projection = torch.matmul(tokens1[t].float(), US.T.float().to(tokens1[t].device)).detach().cpu().numpy()
            # print(projection.shape)
            # plt.figure(0)
            plt.bar([b for b in range(projection.shape[0])], np.abs(projection), label = f'legal_tokenthe' , color = 'b', alpha = 0.1*(t+1))
        for t in range(2):
            projection = torch.matmul(tokens2[t].float(), US.T.float().to(tokens2[t].device)).detach().cpu().numpy()
            # print(projection.shape)
            # plt.figure(0)
            plt.bar([b for b in range(projection.shape[0])], np.abs(projection), label = f'med_tokenthe' , color = 'g', alpha = 0.1*(t+1))
        plt.xlabel('vectors(in groups of five)')
        plt.ylabel('projection')
        plt.legend()
        plt.savefig(f'../0407/token_identity_projection_layer{l}_attention_out.png')
        plt.close()



def show_gate_relation():
    G = torch.load(f"../figures/deepseek/ds_gates.tensors")
    # X = torch.load(f"../figures/deepseek/ds_attention_out_legal.tensors")
    X = torch.load(f"../figures/deepseek/ds_attention_selfout_legal.tensors")

    data = 10
    for layer in range(1,24):
        D = []
        for d in range(data, data+20):
            D += X[layer][d][0][:5]
        D = torch.cat(D, dim = 0)

        U_x, S_x, V_x = torch.svd(D.float().view(-1,2048))
        U_g, S_g, V_g = torch.svd(G[layer][0].detach())
        # print(S_x.shape)
        S_x = S_x/S_x.max()

        

        DOTS = V_x.T @ V_g.to(V_x.device)
        D_MAX,_ = torch.max(DOTS, dim = 1)
        D_SUM = torch.sum(DOTS, dim = 1)
        # print(D_SUM.shape)

        D_MAX = D_MAX/D_MAX.max()
        D_SUM = D_SUM/D_SUM.max()



        plt.figure(0)

        plt.plot(S_x.cpu().numpy(), label = f'EigenValues', alpha = 0.6)
        # plt.plot(D_MAX.cpu().numpy(), label = f'Projection Max', alpha = 0.6)
        plt.plot(D_SUM.cpu().numpy(), label = f'Projection Sum', alpha = 0.3)

        plt.title(f'Layer{layer}')
        plt.xlabel('dims')
        plt.ylabel('evalues')
        plt.legend()
        plt.savefig(f'../0407/ds_eigenvalues_attentionselfout_layer{layer}.png')
        plt.close()

        # COR = V_x.T @ G[layer][0].detach().T.to(V_x.device)
        # plt.figure(0)
        # plt.imshow(COR.abs().T.cpu().numpy(), cmap = 'hot', interpolation = 'nearest', vmin=0, vmax=0.4)
        # plt.colorbar()
        # plt.xlabel('Data EigenVectors')
        # plt.ylabel('Experts')
        # plt.title(f'Layer{layer} w/o res')
        # plt.savefig(f'../0407/ds_eigenmatrix_attentionselfout_layer{layer}.png')
        # plt.close()




def show_expert_relation():
    G = torch.load(f"../figures/deepseek/ds_gates.tensors")
    # X = torch.load(f"../figures/deepseek/ds_attention_out_legal.tensors")
    k = 4
    for layer in range(1,24,5):
        U_g, S_g, V_g = torch.svd(G[layer][0].detach())
        Gate = G[layer][0].detach()
        print(Gate.shape)
        W = torch.load(f"../figures/deepseek/ds_experts{1}_layer{layer}.tensors")
        
        # for e in range(0, 64, 8):
        #     Vs = []
        #     for j in range(8):
        #         U_w, S_w, V_w = torch.svd(W[e+j].float().detach())
        #         Vs.append(V_w.T[:k,:])
        #     VS = torch.cat(Vs)
        #     DOTS = VS @ Gate.T.to(V_w.device)
        #     print(DOTS.shape)
        #     plt.figure(0)
        #     plt.imshow(DOTS.abs().T.cpu().numpy(), cmap = 'hot', interpolation = 'nearest', vmin=0, vmax=0.4)
        #     plt.colorbar()
        #     plt.xlabel(f'Gate_proj EigenVectors({k} per expert)')
        #     plt.ylabel('Gate Vectors')
        #     plt.title(f'Layer{layer} Gate-Expert{e} Relation')
        #     plt.savefig(f'../0415/ds_gate_Gate_proj_relation_layer{layer}_expert{e}.png')
        #     plt.close()
        Vs = []
        for e in range(0, 64):
            
            U_w, S_w, V_w = torch.svd(W[e].float().detach())
            Vs.append(V_w.T[:k,:])
        VS = torch.cat(Vs)
        DOTS = VS @ Gate.T.to(V_w.device)
        plt.figure(0)
        plt.imshow(DOTS.abs().T.cpu().numpy(), cmap = 'hot', interpolation = 'nearest', vmin=0, vmax=0.4)
        plt.colorbar()
        plt.xlabel(f'Gate_proj EigenVectors({k} per expert)')
        plt.ylabel('Gate Vectors')
        plt.title(f'Layer{layer} Gate-Expert{e} Relation')
        plt.savefig(f'../0415/ds_gate_Gate_proj_relation_layer{layer}_all_expert.png')
        plt.close()



        W = torch.load(f"../figures/deepseek/ds_experts{2}_layer{layer}.tensors")
        
        # for e in range(0, 64, 8):
            
        #     Vs = []
        #     for j in range(8):
        #         U_w, S_w, V_w = torch.svd(W[e+j].float().detach())
        #         Vs.append(V_w.T[:k,:])
        #     VS = torch.cat(Vs)
        #     DOTS = VS @ Gate.T.to(V_w.device)
        #     print(DOTS.shape)
        #     plt.figure(0)
        #     plt.imshow(DOTS.abs().T.cpu().numpy(), cmap = 'hot', interpolation = 'nearest', vmin=0, vmax=0.4)
        #     plt.colorbar()
        #     plt.xlabel(f'Up_proj EigenVectors({k} per expert)')
        #     plt.ylabel('Gate Vectors')
        #     plt.title(f'Layer{layer} Gate-Expert{e} Relation')
        #     plt.savefig(f'../0415/ds_gate_Up_proj_relation_layer{layer}_expert{e}.png')
        #     plt.close()
        Vs = []
        for e in range(0, 64):
            U_w, S_w, V_w = torch.svd(W[e].float().detach())
            Vs.append(V_w.T[:k,:])
        VS = torch.cat(Vs)
        DOTS = VS @ Gate.T.to(V_w.device)
        print(DOTS.shape)
        plt.figure(0)
        plt.imshow(DOTS.abs().T.cpu().numpy(), cmap = 'hot', interpolation = 'nearest', vmin=0, vmax=0.4)
        plt.colorbar()
        plt.xlabel(f'Up_proj EigenVectors({k} per expert)')
        plt.ylabel('Gate Vectors')
        plt.title(f'Layer{layer} Gate-Expert{e} Relation')
        plt.savefig(f'../0415/ds_gate_Up_proj_relation_layer{layer}_all_expert.png')
        plt.close()

        W = torch.load(f"../figures/deepseek/ds_experts{3}_layer{layer}.tensors")
        
        # for e in range(0, 64, 8):
        #     Us = []
        #     for j in range(8):
        #         U_w, S_w, V_w = torch.svd(W[e+j].float().detach())
        #         Us.append(U_w.T[:k,:])
        #     US = torch.cat(Us)
        #     DOTS = US @ Gate.T.to(U_w.device)
        #     print(DOTS.shape)
        #     plt.figure(0)
        #     plt.imshow(DOTS.abs().T.cpu().numpy(), cmap = 'hot', interpolation = 'nearest', vmin=0, vmax=0.4)
        #     plt.colorbar()
        #     plt.xlabel(f'Down_proj EigenVectors({k} per expert)')
        #     plt.ylabel('Gate Vectors')
        #     plt.title(f'Layer{layer} Gate-Expert{e} Relation')
        #     plt.savefig(f'../0415/ds_gate_Down_proj_relation_layer{layer}_expert{e}.png')
        #     plt.close()
        Us = []
        for e in range(0, 64):
            U_w, S_w, V_w = torch.svd(W[e].float().detach())
            Us.append(U_w.T[:k,:])
        US = torch.cat(Us)
        DOTS = US @ Gate.T.to(U_w.device)
        print(DOTS.shape)
        plt.figure(0)
        plt.imshow(DOTS.abs().T.cpu().numpy(), cmap = 'hot', interpolation = 'nearest', vmin=0, vmax=0.4)
        plt.colorbar()
        plt.xlabel(f'Down_proj EigenVectors({k} per expert)')
        plt.ylabel('Gate Vectors')
        plt.title(f'Layer{layer} Gate-Expert{e} Relation')
        plt.savefig(f'../0415/ds_gate_Down_proj_relation_layer{layer}_all_expert.png')
        plt.close()


        

        # DOTS = V_x.T @ V_g.to(V_x.device)




        # plt.figure(0)

        # plt.plot(S_x.cpu().numpy(), label = f'EigenValues', alpha = 0.6)
        # # plt.plot(D_MAX.cpu().numpy(), label = f'Projection Max', alpha = 0.6)
        # plt.plot(D_SUM.cpu().numpy(), label = f'Projection Sum', alpha = 0.3)

        # plt.title(f'Layer{layer}')
        # plt.xlabel('dims')
        # plt.ylabel('evalues')
        # plt.legend()
        # plt.savefig(f'../0407/ds_eigenvalues_attentionselfout_layer{layer}.png')
        # plt.close()

        # COR = V_x.T @ G[layer][0].detach().T.to(V_x.device)
        # plt.figure(0)
        # plt.imshow(COR.abs().T.cpu().numpy(), cmap = 'hot', interpolation = 'nearest', vmin=0, vmax=0.4)
        # plt.colorbar()
        # plt.xlabel('Data EigenVectors')
        # plt.ylabel('Experts')
        # plt.title(f'Layer{layer} w/o res')
        # plt.savefig(f'../0407/ds_eigenmatrix_attentionselfout_layer{layer}.png')
        # plt.close()

def new_cluster(X):
    

    # 自定义距离函数（例如内积）
    def custom_distance(x, y):
        return -np.dot(x, y)  # 内积越大，距离越小

    # 定义自定义距离度量
    custom_metric = distance_metric(type_metric.USER_DEFINED, func=custom_distance)

    # 初始化中心点
    initial_centers = kmeans_plusplus_initializer(X, 64).initialize()
    # print(len(initial_centers))

    # 使用 KMeans 和自定义距离
    kmeans_instance = kmeans(X, initial_centers, metric=custom_metric)
    kmeans_instance.process()

    # 获取聚类结果
    clusters = kmeans_instance.get_centers()
    return clusters
    # print(clusters)

def inner_product_clustering(data, n_clusters, max_iter=100, tol=1e-4, device='cpu'):
    """
    基于内积相似度的聚类算法
    
    参数:
        data (torch.Tensor): 输入数据矩阵，形状为 [n_samples, n_features]
        n_clusters (int): 要聚类的数量
        max_iter (int): 最大迭代次数
        tol (float): 收敛阈值（中心点变化小于该值时停止迭代）
        device (str): 计算设备 ('cpu' 或 'cuda')
    
    返回:
        torch.Tensor: 每个样本的聚类标签，形状为 [n_samples]
    """
    # 确保数据在正确的设备上
    data = data.to(device)
    
    # 随机初始化聚类中心
    n_samples, n_features = data.shape
    indices = torch.randperm(n_samples)[:n_clusters]
    centers = data[indices].clone()
    
    for _ in range(max_iter):
        # 计算所有样本与聚类中心的内积相似度
        similarities = torch.mm(data, centers.t())  # [n_samples, n_clusters]
        
        # 分配样本到最近的聚类（内积最大的中心）
        labels = torch.argmax(similarities, dim=1)
        
        # 计算新的聚类中心
        new_centers = torch.zeros_like(centers)
        for k in range(n_clusters):
            # 计算属于当前聚类的所有样本的平均向量
            mask = (labels == k)
            if mask.any():
                new_centers[k] = data[mask].mean(dim=0)
            else:
                # 如果没有样本属于该聚类，则随机重新初始化
                new_centers[k] = data[torch.randint(0, n_samples, (1,))]
        
        # 检查是否收敛
        center_shift = torch.norm(new_centers - centers)
        if center_shift < tol:
            break
            
        centers = new_centers
    
    return centers

def check_transformer(deepseek, data):
    layer_idx = 26
    # attn_output_wo_residual = []
    # with torch.no_grad():
    #     for i, input_ids in enumerate(data):
    #         # print(i)
    #         input_ids = input_ids.to(deepseek.device)

    #         attn_output_wo_residual0 = deepseek(input_ids = input_ids, output_hidden_states=True, output_attentions=True).attentions[layer_idx]
    #         attn_output_wo_residual.append(attn_output_wo_residual0.view(-1, 2048))
    #         print(attn_output_wo_residual0.view(-1, 2048).shape)
    #         if i >=500:
    #             break
    # attn_output_wo_residual = torch.cat(attn_output_wo_residual,dim = 0)
    # # print(attn_output_wo_residual.shape)
    # torch.save(attn_output_wo_residual,f'ds_attn_output_wo_residual_0422.tensors')
    # attn_output_wo_residual = torch.load(f'ds_attn_output_wo_residual_0422.tensors')
    # new_centers = inner_product_clustering(attn_output_wo_residual.float(), 64) # 这里需要用内积而不是 l2 距离进行聚类。tongyi给了一个自定义距离度量的聚类代码在最下面
    # # print(new_centers.shape)
    # new_gate = new_centers
    # # pca = PCA(n_components=64)
    
    # # # 拟合模型并转换数
    # # new_gate = pca.fit_transform(new_centers.cpu().numpy())
    # # 把 center 变成 gate 矩阵就可以了
    # old_gate = deepseek.model.layers[layer_idx].mlp.gate.weight.detach().float()
    # torch.save(old_gate, f'ds_old_gate_0422.tensors')
    # torch.save(new_gate, f'ds_new_gate_0422.tensors')


    # print(new_gate.shape, old_gate.shape)



    old_gate = torch.load(f'ds_old_gate_0422.tensors').to('cuda:1')
    new_gate = torch.load(f'ds_new_gate_0422.tensors').to('cuda:1')

    # # 1 我们的方法：
    I = torch.eye(64).to('cuda:1')
    Q = new_gate @ torch.linalg.pinv(old_gate) # new_gate = Q @ old_gate，Q 的第 i 行就是把原有 64 个 expert 加权起来的权重
    torch.save(Q,f'ds_Q_0422.tensors')
    P = I @ torch.linalg.pinv(Q.T) # new_experts = P @ old_experts, Q.T @ P = I
    # print(P.shape)
    torch.save(P,f'ds_P_0422.tensors')

    # new_experts = P @ [old_experts]

    # # 2 S的方法

    # # 目的：算出在新 moe 中激活 expert i 的数据，它们在旧 expert 里激活哪些奇异方向最大。然后将这些方向对应的矩阵重组出新的 expert i。
    # new_expert_data = [] # 记录新 moe 中每个 expert 会被哪些数据激活
    # for sequence in data:
    #     new_indices = new_gate(sequence) # 这里比如激活 top-6
    #     for new_idx in new_indices:
    #         new_expert_data[new_idx].append(sequence)

    # old_experts, new_experts = [], []
    # for i, new_data in enumerate(new_expert_data):
    #     # new_data 就是会激活新 expert i 的数据，我们想算它在每个旧 expert 里的三种矩阵里对奇异方向的投影值是多少
    #     old_expert_activation_score = [] # old_expert_activation_score[expert_idx].up/gate/down.u/v[singular_vector_idx] = 0
    #     for sequence in new_data:
    #         for token in sequence:
    #             old_expert_indices = old_gate(token) # 它激活了哪几个 expert
    #             for old_expert_idx in old_expert_indices:
    #                 # 对每个它激活的 expert，它对这些 expert 的奇异方向的投影值是多少
    #                 for i, vec in enumerate(old_experts[old_expert_idx].up.v, 1): 
    #                     old_expert_activation_score[old_expert_idx].up.v[i] += torch.dot(token, vec)
    #                     # 这个值是累加的，因为我想算会被送进新 expert i 的所有数据，大家总体上对旧 expert 里奇异方向的投影值如何
    #                     # 这个操作对于 gate.v 和 down.u 也要算一次，也是注意奇异方向的行列
    #                     # 这个操作应该可以通过 torch 的那些矩阵计算简化，应该不用套一万层循环
        
    #     # 有了旧 expert 里每个奇异方向激活值
    #     # 接下来对每个旧 expert，取其中激活值大于 t 的 top-k 个方向，算一个低秩矩阵出来
    #     # 然后再把所有旧 expert 各自被选中方向形成的低秩矩阵取平均，就是新 expert
    #     selected_old_lowrank_experts = []
    #     for old_expert_idx in range(64):
    #         selected_singular_vec_indices = topk(old_expert_activation_score[old_expert_idx].up.v)
    #         old_expert_up = old_experts[old_expert_idx]
    #         # 把选中的方向组合成新矩阵。up 和 gate 算数据激活值的时候只算 v，但重组的时候要用同样位置的 u, s
    #         # 例如，发进新 expert 1 的数据们，在旧 expert 3 的 up.v 的奇异方向里，第 2，5，6 个奇异方向投影值最大
    #         # 那就用 old_expert[3].u[2,5,6] @ old_expert[3].s[2,5,6] @ old_expert[3].v[2,5,6].t() 来重组
    #         # down 矩阵和 up/gate 反过来，算数据激活值的时候只算 u，但是要用 v 和 s 来重组
    #         # 这里选行还是选列可能需要问问大模型确认一下 😭
    #         selected_vecs_matrix = old_expert_up.u[selected_singular_vec_indices] @ old_expert_up.s[selected_singular_vec_indices] @ old_expert_up.v[selected_singular_vec_indices].t()
    #         selected_old_lowrank_experts.append(selected_vecs_matrix)

    #     new_experts[i].up = torch.mean(selected_old_lowrank_experts)
    #     # 同理，gate 和 down 也要算


def extract_all_bad_indices(file_path):
    bad_index_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            bad_line = lines[i]
            if "bad index" in bad_line:
                # 提取方括号里的内容并转为整数列表
                indices = eval(bad_line.split("bad index: ")[-1].strip())
                bad_index_list.append(indices)
    return bad_index_list


def draw_losses():
    plt.figure(0)
    plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
    plt.rcParams['font.size'] = 15
    lossg = torch.load(f'/home/jxzhou/PLM_PER/qwen/0510/losses_good.tensors')
    lossb = torch.load(f'/home/jxzhou/PLM_PER/qwen/0510/losses_bad.tensors')
    plt.plot([i for i in range(len(lossg))],[l.float().cpu().numpy() for l in lossg],label = 'good data loss')
    plt.plot([i for i in range(len(lossb))],[l.float().cpu().numpy() for l in lossb],label = 'bad data loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Data Index')
    plt.savefig(f'/home/jxzhou/PLM_PER/qwen/0510/losses_compare.png')


def draw_losses2():
    plt.figure(0)
    plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
    plt.rcParams['font.size'] = 15
    lossg = torch.load(f'/home/jxzhou/PLM_PER/qwen/0510/losses_good.tensors')
    lossb = torch.load(f'/home/jxzhou/PLM_PER/qwen/0510/losses_bad_ratio.tensors')
    # plt.plot([i*0.05 for i in range(len(lossb))],[lossg[0].float().cpu().numpy() for i in range(len(lossb))],label = 'good data loss')
    
    plt.plot([i*0.05 for i in range(len(lossb))],[l.float().cpu().numpy() for l in lossb])
    plt.ylabel('Loss')
    plt.xlabel('Bad Ratio')
    plt.savefig(f'/home/jxzhou/PLM_PER/qwen/0510/losses_compare.png')

def draw_cluster(highlight_indices):
    for layer in range(1,26,8):
        
        print(layer)

        for d in range(1):
            data1 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0510/hiddenstates_good.tensors')[layer][d].float().view(-1,2048)
            data2 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0510/hiddenstates_bad.tensors')[layer][d].float().view(-1,2048)
            data = torch.cat((data1,data2),dim=0).cpu().numpy()
            u = umap.UMAP(n_components=2).fit(data)
            data1_u = u.transform(data1.cpu().numpy())
            data2_u = u.transform(data2.cpu().numpy())
            plt.figure(0)
            plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
            plt.rcParams['font.size'] = 15
            plt.scatter(data1_u[:,0],data1_u[:,1],label = 'good data embedding', alpha = 0.3)
            plt.scatter(data2_u[:,0],data2_u[:,1],label = 'bad data embedding', alpha = 0.3)
            plt.scatter(data2_u[highlight_indices[d],0],data2_u[highlight_indices[d],1],label = 'bad data case', alpha = 1)
            plt.legend()
            plt.savefig(f'/home/jxzhou/PLM_PER/qwen/0510/embedding_compare_data{d}_layer{layer}.png')
            plt.close()


            # print(data.shape)

def draw_cluster2():

    for layer in range(1,26,4):
        
        print(layer)
        plt.figure(0)
        plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
        plt.rcParams['font.size'] = 15
        data0 = torch.load(f'/home/jxzhou/PLM_PER/qwen/0609/hiddenstates.tensors')[layer]
        data0 = torch.cat(data0,dim = 0).float().cpu().numpy()
        # print(data0.shape)
        u = umap.UMAP(n_components=2).fit(data0)
        for d in range(5):
            # print(d)
            data = torch.load(f'/home/jxzhou/PLM_PER/qwen/0609/hiddenstates.tensors')[layer][d].float().cpu().numpy()
            # print(data.shape)
            
            data = u.transform(data)
            plt.scatter(data[:,0],data[:,1],label = d,alpha=0.3)

        plt.ylabel('umap dim 2')
        plt.xlabel('umap dim 1')
        plt.title(f'token_level_cluster_layer{layer}')
        plt.savefig(f'/home/jxzhou/PLM_PER/qwen/0609/token_level_layer{layer}.png')
        plt.close()





            # plt.figure(0)
            # plt.rcParams['font.family'] = 'DejaVu Math TeX Gyre'
            # plt.rcParams['font.size'] = 15
            # plt.scatter(data1_u[:,0],data1_u[:,1],label = 'good data embedding', alpha = 0.3)
            # plt.scatter(data2_u[:,0],data2_u[:,1],label = 'bad data embedding', alpha = 0.3)
            # plt.scatter(data2_u[highlight_indices[d],0],data2_u[highlight_indices[d],1],label = 'bad data case', alpha = 1)
            # plt.legend()
            # plt.savefig(f'/home/jxzhou/PLM_PER/qwen/0510/embedding_compare_data{d}_layer{layer}.png')
            # plt.close()


            # print(data.shape)



 

if __name__ == "__main__":
    torch.cuda.set_device(1)



    
    torch.set_default_dtype(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("../DeepSeek-16B-2.8B", trust_remote_code=True)
    opens = OpenThoughtsDataset('/home/jxzhou/datasets/Open-Thoughts-114k/default/train',tokenizer)
    print(len(opens))
    # # # # # # # # # # # # # # # ##Qwen1.5-MoE-A2.7B
    # # # # # # # # # # # # # # # ##DeepSeek-16B-2.8B
    # deepseek = AutoModelForCausalLM.from_pretrained('../DeepSeek-16B-2.8B', device_map="auto", trust_remote_code=True)
    
    # # # k1  = deepseek.model.layers[26].mlp.experts[0].gate_proj.weight.data.clone()
    # # # old_gate = torch.load(f'ds_old_gate_0422.tensors').to('cuda:1')
    # # # new_gate = torch.load(f'ds_new_gate_0422.tensors').to('cuda:1')
    # # # P = torch.load(f'ds_P_0422.tensors').to('cuda:1')

    # # # # print([old_gate == new_gate])
    # # # # print(deepseek.model.layers[26].mlp.experts[0].up_proj.weight.data)
    
    # # # deepseek.model.layers[26].transform_Parameters(P, new_gate)
    # # # print(deepseek.model.layers[26].mlp.experts[0].up_proj.weight.data)


    # # # k2  = deepseek.model.layers[26].mlp.experts[0].gate_proj.weight
    # # # print([k1 == k2])

    # test_data = Tokenized_data(['openweb', 'legal', 'med'], tokenizer, total=1000, is_test=True)
    # good = 'bad'
    
    # draw_data_attention(test_data,deepseek, good)

    
    # # # # # # # # # # # # # # # # # # # param_num=  sum(p.numel() for p in deepseek.parameters())

    # # # # # # # ana_emb_distance()
    # # # GATES = [[] for i in range(27)]
    # # EXPERTS = [[] for i in range(27)]
    # # for layer in range(1, 27):
    # #     shared_expert = deepseek.model.layers[layer].mlp.shared_experts
    # #     experts = deepseek.model.layers[layer].mlp.experts
    # # #     gate = deepseek.model.layers[layer].mlp.gate.weight.float()

    # # #     GATES[layer].append(gate)

    # # #     # breakpoint()
    # #     # experts.insert(0, shared_expert)
    # #     experts1 = [expert.gate_proj.weight.float() for expert in experts] # [gate_proj, up_proj, down_proj]
    # #     experts2 = [expert.up_proj.weight.float() for expert in experts] # [gate_proj, up_proj, down_proj]
    # #     experts3 = [expert.down_proj.weight.float() for expert in experts] # [gate_proj, up_proj, down_proj]
   
    # #     # EXPERTS[layer].append(experts1)
    # #     # EXPERTS[layer].append(experts2)
    # #     # EXPERTS[layer].append(experts3)
    # #     print(experts3[1].shape)
    # #     torch.save(experts1, f"../figures/deepseek/ds_experts1_layer{layer}.tensors")
    # #     torch.save(experts2, f"../figures/deepseek/ds_experts2_layer{layer}.tensors")
    # #     torch.save(experts3, f"../figures/deepseek/ds_experts3_layer{layer}.tensors")




    #     # with torch.no_grad():
    #     #     ana_experts(experts, layer)
    # # torch.save(EXPERTS, f"../figures/deepseek/ds_experts.tensors")


    # # # # # ##Synthetic-Persona-Chat
    # # get_clusrer()
    # # get_semantic_groups_compare()

    
    # # draw_icml_fig3('ds_attention_scores_small',50)
    # # draw_icml_fig3('qwen_attention_scores_small',50)
    # # draw_icml_fig4('ds_sentence_embeddings_large')
    # # draw_icml_fig4('qwen_sentence_embeddings_large')


    # # draw_icml_fig3('ds_attention_scores_large',1500)
    # # draw_icml_fig3('qwen_attention_scores_large',1500)

    # # draw_icml_fig2('qwen_sentence_embeddings_locality', 'qwen_input_ids_locality',tokenizer)
    # # draw_icml_fig2('ds_sentence_embeddings_locality', 'ds_input_ids_locality',tokenizer)



    # # ana_emb_distance('ds_attention_with_res')
    # # get_umap('ds_attention_without_res')
    # # get_stablerank_change_sentence()
    # # get_stablerank_change_params()

    # # show_context_preference('legal')
    # # show_token_identity_preference()
    # # show_gate_relation()
    # # show_expert_relation()
    # # get_gate_transformer_Q(10)
    # # check_transformer(0,0)




    # # ana_emb_distance_compare('ds_embeddings_t_shuffle','ds_embeddings')
    # # show_activatation()


    # file_path = f'/home/jxzhou/PLM_PER/qwen/0510/{good}.log'
    # all_bad_indices = extract_all_bad_indices(file_path)
    # # print(all_bad_indices)
    # draw_attention('attentionscores',good,all_bad_indices)
    # draw_losses2()
    # draw_cluster2()