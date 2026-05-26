import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import Wikitxt2ForLM_1103,Wikitxt103ForLM_1103,ACLForLM_1103,Mixdata_1103, AGNewsForLM_1103
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

def get_cluster_centers(input,k =5):
    X = input.mean(axis=1)
    X = X.cpu().numpy()
    kmeans = KMeans(n_clusters=k, n_init= 'auto')
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_
    
    return cluster_centers


def get_outputs_sample(output, raw_output, k =2):

    kmeans = KMeans(n_clusters=k, n_init= 'auto')
    
    # dbscan = DBSCAN(eps=50, min_samples=3)
    # ypred = dbscan.fit_predict(output)
    # print(f'DBSCAN labels: {max(ypred)}')
    # Fit the model to your data
    kmeans.fit(output)

    # Get cluster assignments for each data point
    # labels = kmeans.labels_
    # print(f'cluster_num {labels}')
    # Get the coordinates of the cluster centers
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
    model0 = torch.load('bert-1103-stage0.pth')
    model.bert.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    for i in range(config.num_hidden_layers):
        model.bert.layers.layers[i].load_state_dict(model0.bert.encoders.layers[i].state_dict())
    model.head.load_state_dict(model0.head.state_dict())
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()
    
    # load model checkpoint
    model, optimizer, lr_scheduler, train_loader, val_loader, train_loader2, val_loader2 = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, train_loader2, val_loader2)
    # accelerator.load_state(load_path)
    
    # run once
    model.eval()
    
    all_layer_outputs = [[] for i in range(12)]
    all_layer_outputs_ids = [[] for i in range(12)]

    all_layer_inputs = []
    all_layer_labels = []
    all_layer_attns = []
    #
    all_scores_len = []
    OUT_FOR_PCA = [[] for i in range(13)]
    OUT_FOR_PCA2 = [[] for i in range(13)]

    cluster_centers = [[] for i in range(12)]
    memory = MemoryFromDecoder()
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i <15:          
                print(f"######{i}")                
                mlm_loss, scores, inputs, outputs, expert_ids,att_outs,outputs,ffn_outputs,att_self = model(**batch)
                #
                OUT_FOR_PCA[0].append(model.bert.embeddings(batch['input_ids']))
                scores = memory.output2input(scores)

                
                # scores.to('cpu')
                for j, layer_output in enumerate(layer_outputs):  
                    # layer_output = layer_output.view(config.batch_size,-1)
                    OUT_FOR_PCA[j+1].append(layer_output)

        

    

    with torch.no_grad():
        for i, batch in enumerate(train_loader2):
            if i <15:          
                print(f"######{i}")                
                _, scores, layer_outputs,_ = model(**batch)
                #
                OUT_FOR_PCA[0].append(model.bert.embeddings(batch['input_ids']))
                scores = memory.output2input(scores)
                # scores.to('cpu')
                for j, layer_output in enumerate(layer_outputs):  
                    # layer_output = layer_output.view(config.batch_size,-1)
                    OUT_FOR_PCA[j+1].append(layer_output)

    print(len(OUT_FOR_PCA[0]), len(OUT_FOR_PCA[1]), OUT_FOR_PCA[9][0].shape)

    for kt in range(len(OUT_FOR_PCA)):
        OUT_FOR_PCA[kt] = torch.cat(OUT_FOR_PCA[kt], dim =0 )
    print(len(OUT_FOR_PCA[0]), len(OUT_FOR_PCA[1]), OUT_FOR_PCA[9][0].shape)


    for kt in range(len(OUT_FOR_PCA)):
        PCA = pca(OUT_FOR_PCA[kt])
        plt.figure(kt)
        plt.scatter(PCA[:960, 0], PCA[:960, 1], c = 'r', cmap=plt.cm.Set1)
        plt.scatter(PCA[960:, 0], PCA[960:, 1], c = 'b', cmap=plt.cm.Set1)
        
        plt.title('PCA Dimensionality Reduction')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig('1104-PCA-Dimensionality-Reduction-mixdata-agnews'+'%d.png'%kt)
        plt.show()




    # accelerator.print(f'Number of Samples batches: {len(all_layer_outputs[0])}')
    
    # calculate pca




if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/bert.json')
    dataset = Mixdata_1103(config=config)
    new_dataset = AGNewsForLM_1103(config)
    
    model = base_models.BertWithSavers(config=config)
    # model = base_models.BertWithDecoders(config=config)
    # model = nn.DataParallel(model)
    
    load_path = "./output-formal-1X"
    layer_pca(model=model, dataset=dataset, new_dataset=new_dataset)
