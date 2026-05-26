import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import MoMoE_longtailed,Wikipedia,MoMoE_MIXED_LEGAL_REVIEW,MoMoE_FEWER_SPECIFIC_GENERAL,MoMoE_MIXED_LEGAL_PUBMED,MoMoE_MIXED_WARMUP,MoMoE_WIKI103_WARMUP,MixedData_0110_0,MixedData_1211_0,RestaurantForLM_small, ACLForLM_small, MixedData, MixedData_1121,Wikitxt103ForLM_0102_warmup
from accelerate import Accelerator,load_checkpoint_and_dispatch
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch.optim as optim
from transformer.Transformer import MemoryFromDecoder
import torch
import numpy as np
import random
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors





# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

from sklearn.metrics import pairwise_distances




def plot_k_distance_graph(X, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)
    k_distances = distances[:, k-1]
    k_distances = np.sort(k_distances)
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.ylabel(f'{k}-Distance')
    plt.xlabel('Points sorted by distance')
    plt.title(f'k-Distance Graph for k={k}')
    plt.grid(True)
    plt.show()
    return k_distances

def dbscan_grid_search(X, eps_range, min_samples_range):
    best_score = -1
    best_params = {'eps': None, 'min_samples': None}
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            labels = db.labels_
            if len(set(labels)) > 1:  # 至少有两个簇
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
    return best_params, best_score








# def plot_k_distance_graph(X, k,l):
#     """
#     绘制 k-距离图，用于选择 DBSCAN 中的 eps 参数。
    
#     参数:
#     X (numpy.ndarray): 数据集，形状为 (n_samples, n_features)。
#     k (int): k 值，通常等于 DBSCAN 中的 min_samples。
#     """
#     # 计算每个点到其 k 近邻的距离
#     nbrs = NearestNeighbors(n_neighbors=k).fit(X)
#     distances, indices = nbrs.kneighbors(X)
    
#     # k-距离指的是每个点到其第 k 近邻的距离
#     k_distances = distances[:, k-1]
    
#     # 将 k-距离排序
#     k_distances = np.sort(k_distances)
    
#     # 绘制 k-距离图
#     plt.figure(l+100)
#     plt.plot(k_distances)
#     plt.ylabel(f'{k}-Distance')
#     plt.xlabel('Points sorted by distance')
#     plt.title(f'k-Distance Graph for k={k}')
#     plt.grid(True)
#     plt.show()
#     plt.savefig('0516-DBSCAN-EPS-l%d.png'%l)
def cal_cluster_center(embeddings, labels):
    labelwise_embeddings = []
    for label in set(labels):
        labelwise_embeddings.append([])
    for i in range(len(labels)):
        labelwise_embeddings[labels[i]].append(embeddings[i])
    centers = torch.zeros(len(labelwise_embeddings),128)
    for i in range(len(labelwise_embeddings)):
        labelwise_embeddings[i] = np.array(labelwise_embeddings[i])
        centers[i,:] =torch.tensor(labelwise_embeddings[i].mean(0))

    radius = torch.zeros(len(labelwise_embeddings),1)
    for i in range(len(labelwise_embeddings)):

        radius[i,:]=torch.tensor(np.linalg.norm(torch.tensor(labelwise_embeddings[i]) - centers[i], axis=1).mean())

    return centers, radius

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def get_available_cuda_device() -> int:
    max_devs = torch.cuda.device_count()
    for i in range(max_devs):
        try:
            mem = torch.cuda.mem_get_info(i)
        except:
            continue
        if mem[0] / mem[1] > 0.85:
            return i
    return -1


def get_gradient_norms(model):
    """Utility function to get gradient norms of a model."""
    return [param.grad.norm().item() for param in model.parameters() if param.grad is not None]

def pca(input, threshold=0.80):
    X = input.mean(axis=1)
    X = X.cpu().numpy()
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    # pca = PCA(n_components=X.shape[1])
    
    # explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
    # num_components = np.argmax(explained_variance_ratio >= threshold) + 1

    pca = PCA(n_components=2)
    X_pca_efficient = pca.fit_transform(X_std)    
    X_pca_efficient = torch.tensor(X_pca_efficient) 


    
    return X_pca_efficient


def get_PCA_obj(inputs, threshold = 0.9):
    # embeddings = [N, embed_dim]
    print('create PCA object')
    inputs = inputs.mean(axis = 1)
    inputs = inputs.cpu().numpy()
    pca = PCA()
    pca.fit(inputs)

    # find the first k components that in total explain threshold of the variance
    unique_dims = 0
    total = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        total += var
        if total >= threshold:
            unique_dims = i
            break

    # print(f'pca unique dims: {unique_dims},  {pca.explained_variance_ratio_[:unique_dims]}')
    # print(torch.tensor(pca.transform(inputs)[:,:unique_dims]).shape)
    return torch.tensor(pca.transform(inputs)[:,:unique_dims+1]), pca, unique_dims

def differentiable_pca(x, k=2):
    scaler = StandardScaler()
    standarlized_x = scaler.fit_transform(x.cpu().numpy())
    x = torch.from_numpy(standarlized_x)
    x = x.to('cuda')
    # Perform SVD

    pca = PCA(n_components=k)
    pca.fit(x.cpu().numpy())  
    U, S, V = torch.svd(x)

    # Extract the top k principal components
    principal_components = U[:, :k]

    # print(f'pca.explained_variance_ratio_sum {pca.explained_variance_ratio_.sum()}')
    # print(f'pca.explained_variance_ratio_ {pca.explained_variance_ratio_}')
    # print(f'pca.explained_variance_ {pca.explained_variance_}')
    # Project data onto these components
    # reduced_data = x @ V[:, :k]
    # print(f'pca.fit_transform(x) {pca.fit_transform(x.cpu().numpy()).shape}')
    # return reduced_data
    return pca.fit_transform(x.cpu().numpy())

def get_cluster_centers(input,k = 5,l =0):
    X = input.mean(axis=1)
    # X = input.view(input.shape[0],-1)
    lora_mat_1 = torch.load('0514-lora_mat.pth')
    X = torch.matmul(X, lora_mat_1.to(device)).cpu().numpy()

    # X_O = X.cpu().numpy()
    # print(X_O.shape)
    kmeans_0 = KMeans(n_clusters=k, n_init = "auto")
    kmeans_0.fit(X)
    original_centers0 = kmeans_0.cluster_centers_
    overall_center = np.mean(original_centers0, axis=0)

    labels = kmeans_0.labels_

    # distances = pairwise_distances(X_O, overall_center)

    # # 找到距离中的指定百分位数
    # distance_threshold = np.percentile(distances,99)

    # # 选择距离大于阈值的点
    # filtered_data = X_O[distances.flatten() > distance_threshold]

    # # 对筛选出的数据进行新的聚类
    # new_kmeans = KMeans(n_clusters=4, random_state=42).fit(filtered_data)


    # nnew_centers = new_kmeans.cluster_centers_
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X_O)




    # # 手动初始化聚类中心
    # num_clusters = k  # 选择的聚类数目
    # kmeans_1 = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)
    # kmeans_1.fit(X)
    # farthest,_ = pairwise_distances_argmin_min(kmeans_1.cluster_centers_, X)
    # print(farthest)
    # init_centers = X[farthest]

    # # 使用手动初始化的中心进行聚类
    # kmeans = KMeans(n_clusters=num_clusters, init=init_centers, n_init=1, random_state=0)
    # # kmeans = KMeans(n_clusters=k, n_init= 'auto')
    # kmeans.fit(X)
    # # X = kmeans.transform(X)
    # X = scaler.inverse_transform(X)
    # # original_centers = scaler.inverse_transform(kmeans_0.cluster_centers_)
    # # cluster_centers = kmeans.cluster_centers_


    

    # # 计算新的聚类中心
    # new_centers = []
    # for center in original_centers0:
    #     # 计算移动方向（从聚类中心指向整体中心）
    #     direction = overall_center - center
    #     direction /= np.linalg.norm(direction)  # 单位化方向向量
    #     new_center = center - direction * 5  # 举例，沿着方向移动10%
    #     new_centers.append(new_center)

    # new_centers = torch.tensor(new_centers)
    # plt.figure(l)
    # plt.scatter(X_O[:,0],X_O[:,1])
    # plt.scatter(nnew_centers[:,0],nnew_centers[:,1],color = "r")
    # plt.savefig("0120-oldcluster-test-l%d.png"%l)
    plt.figure(l+100)
    plt.scatter(X[:,0],X[:,1])
    plt.scatter(original_centers0[:,0],original_centers0[:,1],color = "r")
    plt.savefig("0120-cluster-test-l%d.png"%l)

    return torch.tensor(original_centers0)

def get_cluster_centers_for_MoMoE(input,k = 2,e = 2):
    X = input.mean(axis=1)
    X = X.cpu().numpy()
    kmeans = KMeans(n_clusters=k, n_init= 'auto')
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers)
    labels = kmeans.labels_
    sub_centers = {}

    for i in range(e):  # 对于两个主簇
        sub_data = X[labels == i]
        kmeans_sub = KMeans(n_clusters=2, random_state=0)
        kmeans_sub.fit(sub_data)
        sub_centers[i] = kmeans_sub.cluster_centers_

    
    return [cluster_centers,sub_centers]

def get_special_cluster_centers(input,k = 5):
    X = input.mean(axis=1)
    X = X.cpu().numpy()
    kmeans = KMeans(n_clusters=k, n_init= 'auto')
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers)
    return cluster_centers

def get_outputs_sample(output, raw_output, k =2):

    kmeans = KMeans(n_clusters=k, n_init= 'auto')

    kmeans.fit(output)


    cluster_centers = kmeans.cluster_centers_
    # print(f'cluster_centers {cluster_centers}')


    cluster_assignments = kmeans.labels_

    # Initialize an empty dictionary to store data points for each cluster
    clustered_data = {cluster_id: [] for cluster_id in range(k)}
    clustered_ids = {cluster_id: [] for cluster_id in range(k)}

    # Organize the raw data into clusters
    for data_point, cluster_id in zip(output, cluster_assignments):
        clustered_data[cluster_id].append(data_point)
        ids = np.argwhere(output == data_point)
        clustered_ids[cluster_id].append(ids[0,0])

    # print(f'ids_cluster{clustered_ids}')
    output_to_saved =[]
    IDS = []
    for i in range(len(cluster_centers)):
        num_of_percluster = len(clustered_data[i])
        avge = np.sum((clustered_data[i]-cluster_centers[i])**2)/num_of_percluster
        # print(f'avge{avge}')
        for l in range(len(clustered_data[i])):
            cs = clustered_data[i][l]
            index = clustered_ids[i][l]
            
            # print(f'avge per{np.sum((cs-cluster_centers[i])**2)}')
            if np.sum((cs-cluster_centers[i])**2) <= avge:
                # print(f'ids {index}')
                output_to_saved.append(raw_output[index])
                IDS.append(index)
                # print(f'output.shape:{raw_output[index].shape}')

    # print(f'output_to_saved.shape:{torch.tensor([item.cpu().detach().numpy() for item in output_to_saved]).cuda().shape}')
    # Print the clustered data
    # for cluster_id, data_points in clustered_data.items():
    #     print(f"Cluster {cluster_id + 1}:")
    #     for point in data_points:
    #         print(point)
    #     print()
    
    return IDS

def get_new_output(output):
    ase = torch.zeros(output.shape[0],output.shape[2])
    for i in range(output.shape[0]):
        for j in range(output.shape[2]):
            ase[i][j] = torch.mean(output[i,:,j])

    return ase

def layer_pca(model, dataset, new_dataset):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader2, val_loader2 = new_dataset.train_loader, new_dataset.val_loader
    num_updates = 70 * len(train_loader)
    model01 = base_models.BertForMLM(config=config)
    # model0 = torch.load('0311-BERT-FOR-CLUSTER-MIXED_LEGAL_REVIEW.pth')
    load_checkpoint_and_dispatch(model01, 'MODELS_0514/BERT-768ffns-mixed-small-1.5e-4',device_map={"": device}) 
    # model.load_state_dict(model01.state_dict())

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()
    model.bert.embeddings.load_state_dict(model01.bert.embeddings.state_dict())
    for i in range(config.num_hidden_layers):
        model.bert.layers.layers[i].load_state_dict(model01.bert.encoders.layers[i].state_dict())
    model.head.load_state_dict(model01.head.state_dict())
    # load model checkpoint
    model, optimizer, lr_scheduler, train_loader, val_loader, train_loader2, val_loader2 = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, train_loader2, val_loader2)
    # accelerator.load_state(load_path)
    
    # run once
    model.eval()

    out_for_cluster = [[] for i in range(12)]
    cluster_centers = []
    out_for_cluster_special = [[] for i in range(12)]
    special_cluster_centers = []
    cluster_radius = []

    # lora_mat = torch.randn([768, 128])
    # torch.save(lora_mat,'0514-lora_mat.pth')

    
    


    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i>=900:
                break
            if i %30 == 0:          
                print(f"######{i}")                
                _, _,_,layer_outputs = model(**batch)
                #
               
                # out_for_cluster[0].append(model.bert.embeddings(batch['input_ids']).detach())
                # out_for_cluster_special[0].append(model.bert.embeddings(batch['input_ids']).detach())
                
                # scores.to('cpu')
                for j, layer_output in enumerate(layer_outputs[:]):  
                    # layer_output = layer_output.view(config.batch_size,-1)
                    
                    # out_for_cluster_special[j+1].append(layer_output)
                    
                    out_for_cluster[j].append(layer_output)
    for j in range(len(out_for_cluster)):
        out_for_cluster[j] = torch.cat(out_for_cluster[j], dim =0 ).detach()
        # print(out_for_cluster[j].shape)
    # PRO_VECS = []
    # for l in range(config.num_hidden_layers):
    #     eig_vecs0 = torch.load('0311-layer%d_pro_vec-3t-MIXED_LARGE.pth'%l, map_location='cuda')
    #     PRO_VECS.append(eig_vecs0)
    for j in range(len(out_for_cluster)):
        # lora_mat_1 = torch.load('0514-lora_mat.pth')
        # X = torch.matmul(out_for_cluster[j].mean(1), lora_mat_1.to(device)).cpu().numpy()

        # k = 4
        # k_distances = plot_k_distance_graph(X, k)

        # # 选择 eps 的范围（从图中手动选择范围）
        # eps_range = np.arange(0.1, 1.0, 0.1)
        # min_samples_range = range(2, 10)

        # # 进行网格搜索
        # best_params, best_score = dbscan_grid_search(X, eps_range, min_samples_range)

        # # 输出最佳参数
        # print("Best parameters found: ", best_params)
        # print("Best silhouette score: ", best_score)

        # out_for_cluster[j] = torch.matmul(out_for_cluster[j],PRO_VECS[j])[:,:,-config2.unique_hidden_size:]
        # print(out_for_cluster[j].shape)
        # plot_k_distance_graph(X,4,j)
        # clust_alg = DBSCAN(eps=55, min_samples=8).fit(X)
        # clust_alg = KMeans(n_clusters=clust_num, random_state=0, n_init=10).fit(embeddings)
        # print(clust_alg.labels_)
        # print(max(clust_alg.labels_) + 1)

        # XU = umap.UMAP(random_state=42).fit_transform(X)
    
        # plt.figure(j)
        # plt.scatter(XU[:,0],XU[:,1],c = clust_alg.labels_)
        # plt.savefig("0516-cluster-DBSCAN-bert-l%d.png"%j)

        # centers, radius = cal_cluster_center(X, clust_alg.labels_)
        # print(radius.shape)
        # # centers= torch.cat(centers)
        
        # cluster_centers.append(centers)
        # cluster_radius.append(radius)
        
        cluster_centers.append(get_cluster_centers(out_for_cluster[j], k = config2.num_transformer,l=j))
    # for j in range(len(out_for_cluster_special)):
    #     out_for_cluster_special[j] = torch.cat(out_for_cluster_special[j], dim =0 )
    #     print(out_for_cluster_special[j].shape)
    
    # for j in range(len(out_for_cluster_special)):
    #     out_for_cluster_special_with_pca, pca, uni_dim = get_PCA_obj(out_for_cluster_special[j])
    #     special_cluster_centers.append(get_cluster_centers(out_for_cluster_special_with_pca, k = config.num_experts))
    
    # print(torch.tensor(pca.transform(out_for_cluster_special[j].view(-1,768).cpu().numpy())).shape)
        # print(get_cluster_centers(out_for_cluster[j]).shape)
    # print(cluster_centers[0].shape, cluster_centers[5].shape, cluster_centers[11].shape)
    
    # out_for_cluster_special_with_pca, uni_dim = get_PCA_obj(out_for_cluster_special)
    # special_cluster_centers = get_special_cluster_centers(out_for_cluster_special[:,:,:uni_dim], k = config.num_experts)
    # print(special_cluster_centers.shape)
    layer_centers = {}
    # layer_radius = {}
    # special_layer_centers = {}
    for i, layer in enumerate(cluster_centers):
        layer_centers['layer' + str(i+1) ] = cluster_centers[i]
    # for i, layer in enumerate(cluster_radius):
    #     layer_radius['layer' + str(i+1) ] = cluster_radius[i]
    # for i, layer in enumerate(special_cluster_centers):
    #     special_layer_centers['layer' + str(i+1) ] = special_cluster_centers[i]
    torch.save(layer_centers, '0518-layer_centers-ffn-4t-longtail.pth')
    # torch.save(layer_radius, '0517-layer_radius-3t-longtail.pth')


    # torch.save(special_layer_centers, 'special_layer_centers.pth')

    




    # accelerator.print(f'Number of Samples batches: {len(all_layer_outputs[0])}')
    
    # calculate pca




if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/bert.json')
    config2 = BertConfig.from_json_file('config/MoMoE.json')

    dataset = MoMoE_longtailed(config=config)
    new_dataset = MoMoE_MIXED_LEGAL_REVIEW(config)
    torch.cuda.set_device(6)
    device = torch.device("cuda:6")

    model = base_models.BertWithSavers(config=config)
    # model = base_models.BertWithDecoders(config=config)
    # model = nn.DataParallel(model)
    # 
    # load_path = "./output-formal-1X"
    layer_pca(model=model, dataset=dataset, new_dataset=new_dataset)
