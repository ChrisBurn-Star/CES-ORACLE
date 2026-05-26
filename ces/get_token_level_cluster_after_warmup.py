import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import MixedData_0110_0,MixedData_1211_0,RestaurantForLM_small, ACLForLM_small, MixedData, MixedData_1121,Wikitxt103ForLM_0102_warmup
from accelerate import Accelerator
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

def get_cluster_centers(input,k = 5):
    X = input.view(-1,input.shape[-1])
    X = X.cpu().numpy()
    kmeans = KMeans(n_clusters=k, n_init= 'auto')
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers)
    return cluster_centers

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
    model0 = torch.load('0108-bert-MixedData_1211-5000.pth')
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()
    model.bert.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    for i in range(config.num_hidden_layers):
        model.bert.layers.layers[i].load_state_dict(model0.bert.encoders.layers[i].state_dict())
    model.head.load_state_dict(model0.head.state_dict())
    # load model checkpoint
    model, optimizer, lr_scheduler, train_loader, val_loader, train_loader2, val_loader2 = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, train_loader2, val_loader2)
    # accelerator.load_state(load_path)
    
    # run once
    model.eval()

    out_for_cluster = [[] for i in range(12)]
    cluster_centers = []
    out_for_cluster_special = [[] for i in range(12)]
    special_cluster_centers = []
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i %10 == 0:          
                print(f"######{i}")                
                _, _, layer_outputs,_ = model(**batch)
                #
               
                out_for_cluster[0].append(model.bert.embeddings(batch['input_ids']))
                out_for_cluster_special[0].append(model.bert.embeddings(batch['input_ids']))
                
                # scores.to('cpu')
                for j, layer_output in enumerate(layer_outputs[:-1]):  
                    # layer_output = layer_output.view(config.batch_size,-1)
                    
                    # out_for_cluster_special[j+1].append(layer_output)
                    
                    out_for_cluster[j+1].append(layer_output)
    for j in range(len(out_for_cluster)):
        out_for_cluster[j] = torch.cat(out_for_cluster[j], dim =0 )
        # print(out_for_cluster[j].shape)
    for j in range(len(out_for_cluster)):
        cluster_centers.append(get_cluster_centers(out_for_cluster[j], k = config.num_experts))
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
    special_layer_centers = {}
    for i, layer in enumerate(cluster_centers):
        layer_centers['layer' + str(i+1) ] = cluster_centers[i]
    # for i, layer in enumerate(special_cluster_centers):
    #     special_layer_centers['layer' + str(i+1) ] = special_cluster_centers[i]
    torch.save(layer_centers, '0108-layer_centers-2expert-MixedData_1211-5000-token.pth')
    # torch.save(special_layer_centers, 'special_layer_centers.pth')

    




    # accelerator.print(f'Number of Samples batches: {len(all_layer_outputs[0])}')
    
    # calculate pca




if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/bert.json')
    dataset = MixedData_0110_0(config=config)
    new_dataset = ACLForLM_small(config)
    
    model = base_models.BertWithSavers(config=config)
    # model = base_models.BertWithDecoders(config=config)
    # model = nn.DataParallel(model)
    # 
    # load_path = "./output-formal-1X"
    layer_pca(model=model, dataset=dataset, new_dataset=new_dataset)
