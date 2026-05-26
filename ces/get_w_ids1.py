import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import PubMedForLM,LegalForLM,Overruling,GAD,Casehold,GLUE,EUADR,multi_domains,Wikipedia_forids,MoMoE_MIXED_LEGAL_REVIEW,MoMoE_FEWER_SPECIFIC_GENERAL,MoMoE_MIXED_LEGAL_PUBMED,MoMoE_MIXED_WARMUP,MoMoE_WIKI103_WARMUP,MixedData_0110_0,MixedData_1211_0,RestaurantForLM_small, ACLForLM_small, MixedData, MixedData_1121,Wikitxt103ForLM_0102_warmup
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

# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min



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
    X_O = X.cpu().numpy()
    kmeans_0 = KMeans(n_clusters=k, n_init = "auto")
    kmeans_0.fit(X_O)
    original_centers0 = kmeans_0.cluster_centers_



    scaler = StandardScaler()
    X = scaler.fit_transform(X_O)




    # 手动初始化聚类中心
    num_clusters = k  # 选择的聚类数目
    kmeans_1 = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)
    kmeans_1.fit(X)
    farthest,_ = pairwise_distances_argmin_min(kmeans_1.cluster_centers_, X)
    print(farthest)
    init_centers = X[farthest]

    # 使用手动初始化的中心进行聚类
    kmeans = KMeans(n_clusters=num_clusters, init=init_centers, n_init=1, random_state=0)
    # kmeans = KMeans(n_clusters=k, n_init= 'auto')
    kmeans.fit(X)
    # X = kmeans.transform(X)
    X = scaler.inverse_transform(X)
    original_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    # cluster_centers = kmeans.cluster_centers_


    overall_center = np.mean(original_centers, axis=0)

    # 计算新的聚类中心
    new_centers = []
    for center in original_centers:
        # 计算移动方向（从聚类中心指向整体中心）
        direction = overall_center - center
        direction /= np.linalg.norm(direction)  # 单位化方向向量
        new_center = center - direction * 0  # 举例，沿着方向移动10%
        new_centers.append(new_center)

    new_centers = torch.tensor(new_centers)
    plt.figure(l)
    plt.scatter(X_O[:,0],X_O[:,1])
    plt.scatter(original_centers0[:,0],original_centers0[:,1],color = "r")
    plt.savefig("0120-oldcluster-test-l%d.png"%l)
    plt.figure(l+100)
    plt.scatter(X[:,0],X[:,1])
    plt.scatter(new_centers[:,0],new_centers[:,1],color = "r")
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
def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    return layer_data
def layer_pca(model, dataset, name):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader



######finetune#####
    # val_loader1,val_loader2,val_loader3,val_loader4 = dataset.val_loader1,dataset.val_loader2,dataset.val_loader3,dataset.val_loader4
    # train_loader1,train_loader2,train_loader3,train_loader4 = dataset.train_loader1,dataset.train_loader2,dataset.train_loader3,dataset.train_loader4


    num_updates = 70 * len(train_loader)
    model01 = base_models.BertForMLM(config=config)
    # model0 = torch.load('0311-BERT-FOR-CLUSTER-MIXED_LEGAL_REVIEW.pth')
    load_checkpoint_and_dispatch(model01, 'MODELS_0429/0429-BERT-768ffns-wikipedia-1.5e-4',device_map={"": device}) 
    model.load_state_dict(model01.state_dict())
    cluster_centers = load_layer_data('0429-layer_centers-4t-longtail.pth')
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()

    # load model checkpoint
    model, optimizer, lr_scheduler, train_loader,val_loader= accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)

    # model, optimizer, lr_scheduler, train_loader, train_loader1,train_loader2,train_loader3,train_loader4 = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, train_loader1,train_loader2,train_loader3,train_loader4)
    # accelerator.load_state(load_path)
    model.to(device)
    # val_loaders = [val_loader1,val_loader2,val_loader3,val_loader4]
    # train_loaders = [train_loader1,train_loader2,train_loader3,train_loader4]

    
    # run once
    model.eval()
    # SPLIT = t*4520000
    # SPLIT2 = (t+1)*11300
    

    W_ids = torch.zeros(1500000).long()
    W_ids1 = torch.zeros(1000000).long()

    with torch.no_grad():
        for i, batch in enumerate(train_loader):

            

            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            _,_,_,_,_,_,_,wids = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers[-1])
            W_ids[config.batch_size*i:config.batch_size*i+batch['input_ids'].shape[0]] = wids.detach()
            # print(i, W_ids[config.batch_size*i:config.batch_size*i+batch['input_ids'].shape[0]])
            if (i+1)*config.batch_size >= 1000000:break
    print("train_set finish")
    torch.save(W_ids, '0503-W_IDS-%s-T.pth'%name)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # if SPLIT<=i and i <SPLIT2:
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            _,_,_,_,_,_,_,wids = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers[-1])
            W_ids1[config.batch_size*i:config.batch_size*i+batch['input_ids'].shape[0]] = wids.detach()
            # print(i, W_ids1[config.batch_size*i:config.batch_size*i+batch['input_ids'].shape[0]])
            if (i+1)*config.batch_size >= 50000:break

    torch.save(W_ids1, '0503-W_IDS-%s-V.pth'%name)
    



if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/bert.json')
    config2 = BertConfig.from_json_file('config/MoMoE.json')
    t = 6
    # dataset = Wikipedia_forids(config=config,t=t)
    datasets = {}
    # datasets['pubmed'] = PubMedForLM(config)
    datasets['legal'] = LegalForLM(config)

    # datasets['casehold'] = Casehold(config)

    # datasets['overruling'] = Overruling(config)

    # datasets['sst2'] = GLUE(config)

    # datasets['gad'] = GAD(config)
    # datasets['euadr'] = EUADR(config)

    # new_dataset = MoMoE_MIXED_LEGAL_REVIEW(config)
    torch.cuda.set_device(t)
    device = torch.device("cuda")
    model = base_models.BertForMLM_toshow(config=config)
    # model = base_models.BertWithDecoders(config=config)
    # model = nn.DataParallel(model)
    # 
    # load_path = "./output-formal-1X"
    for i in datasets:
        layer_pca(model=model, dataset=datasets[i], name = i)
