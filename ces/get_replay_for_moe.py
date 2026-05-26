import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import RestaurantForLM_small, ACLForLM_small, MixedData, MixedData_stage1, old_MixedData_after_stage1, Mixdata_1115,ACLForLM_1103, MixedData_1121
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


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    return layer_data

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
    cluster_centers = torch.tensor(cluster_centers)
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

def get_the_other_routes_ids(expert_ids,config):
    IDS = []
    for i in range(expert_ids[0].shape[0]):
        c = 0
        for j in range(len(expert_ids)):
            if expert_ids[j][i] == 1:
                t = 0
            else:
                t = 1
            c+=t*2**j
        IDS.append(c)
    return IDS


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    return layer_data
def get_routes_ids(expert_ids,config):
    IDS = []
    for i in range(expert_ids[0].shape[0]):
        c = 0
        for j in range(len(expert_ids)):
            c+=expert_ids[j][i]*2**j
        IDS.append(c)
    return IDS


def layer_pca(model, old_dataset,new_dataset, router):
    train_loader, val_loader = old_dataset.train_loader, old_dataset.val_loader
    
    train_loader2, val_loader2 = new_dataset.train_loader, new_dataset.val_loader
    num_updates = 70 * len(train_loader)
    model = torch.load('1120_rose_model_satge1.pth')
    # model.load_state_dict(model0.state_dict())
    old_cluster_centers = load_layer_data('1121-layer_centers-2expert.pth')
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()
    model0 = torch.load('1115-only-bert-formoe-stage0.pth')
    router.bert.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    for i in range(config.num_hidden_layers):
        router.bert.layers.layers[i].load_state_dict(model0.bert.encoders.layers[i].state_dict())
    router.head.load_state_dict(model0.head.state_dict())
    
    # load model checkpoint
    model, router, optimizer, lr_scheduler, train_loader, val_loader, train_loader2, val_loader2 = accelerator.prepare(model, router,optimizer, lr_scheduler, train_loader, val_loader, train_loader2, val_loader2)
    # accelerator.load_state(load_path)
    
    # run once
    model.eval()

    out_for_cluster = [[] for i in range(12)]
    cluster_centers = []
    outputs = [[[]for j in range(config.num_experts)] for i in range(12)]
    inputs = [[[]for j in range(config.num_experts)] for i in range(12)]
    attens = [[[]for j in range(config.num_experts)] for i in range(12)]
    labels = [[[]for j in range(config.num_experts)] for i in range(12)]
    fake_data = [[[]for j in range(config.num_experts)] for i in range(12)]
    routes = [[] for i in range(2**12)]
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i %5 == 0:     
   
                hidden_states_for_router = []

                _, _, layer_outputs,_ = router(**batch)
                    #
                
                hidden_states_for_router.append(router.bert.embeddings(batch['input_ids']))
                hidden_states_for_router = hidden_states_for_router  + layer_outputs[0:-1]             
                _, _, _, layer_outputs, experts_ids = model(batch['input_ids'],batch['attention_mask'], batch['labels'], old_cluster_centers, hidden_states_for_router)
                #
                # print(experts_ids)
                IDS = get_routes_ids(experts_ids,config)
                # print(IDS)
                for kl in range(len(IDS)):
                    routes[IDS[kl]].append({'input_ids':batch['input_ids'][kl,:], 'attention_mask': batch['attention_mask'][kl,:], 'labels':batch['labels'][kl,:]})
                    
                # print(routes[c])
                # # print(experts_ids)
                # for i in range(len(outputs)):
                #     for j in range(len(outputs[0])):
                #         if len(layer_outputs[i][j]):
                #             outputs[i][j].append(layer_outputs[i][j][0][:32])
                #             fake_data[i][j].append(batch)
                # for i in range(len(inputs)):
                #     for j in range(len(inputs[0])):
                #         if len(layer_inputs[i][j]):
                #             inputs[i][j].append(layer_inputs[i][j][0][:32])
                #             attens[i][j].append(batch['attention_mask'][:32])
                #             labels[i][j].append(batch['labels'][:32])
                # print(layer_outputs[0][3][0])
                # print(layer_outputs[0][3][0].shape)
                # del layer_inputs, layer_outputs, batch
                # torch.cuda.empty_cache()
            
        # print(outputs[0][3][0])
        print('***')

        # for k in range(len(routes)):
        #     if len(routes[k]):
                
        #         print(routes[k])
        # for i in range(len(outputs)):
        #     for j in range(len(outputs[0])):
        #         if len(outputs[i][j]):
        #             # print(i,j)
        #             outputs[i][j] = torch.cat(outputs[i][j],dim =0)
        #             # print(outputs[i][j].shape)
        # for i in range(len(inputs)):
        #     for j in range(len(inputs[0])):
        #         if len(inputs[i][j]):
        #             # print(i,j)
        #             inputs[i][j] = torch.cat(inputs[i][j],dim =0)
        # for i in range(len(labels)):
        #     for j in range(len(labels[0])):
        #         if len(labels[i][j]):
        #             # print(i,j)
        #             labels[i][j] = torch.cat(labels[i][j],dim =0)
        # for i in range(len(attens)):
        #     for j in range(len(attens[0])):
        #         if len(attens[i][j]):
        #             # print(i,j)
        #             attens[i][j] = torch.cat(attens[i][j],dim =0)
                    # print(outputs[i][j].shape)
        # outputs[0][1] = torch.cat(outputs[0][1],dim =0)
        # print(outputs[0][1].shape)
    
        
    # layer_replays = {}
    # input_layer_replays = {}
    # atten_layer_replays = {}
    # label_layer_replays = {}
    # fake_data_replays = {}
    replay_data = {}
    # for i, layer in enumerate(outputs):
    #     layer_replays['layer' + str(i+1) ] = outputs[i]
    for i, layer in enumerate(routes):
        replay_data['layer' + str(i+1) ] = routes[i]
    # for i, layer in enumerate(inputs):
    #     input_layer_replays['layer' + str(i+1) ] = inputs[i]
    # for i, layer in enumerate(attens):
    #     atten_layer_replays['layer' + str(i+1) ] = attens[i]
    # for i, layer in enumerate(labels):
    #     label_layer_replays['layer' + str(i+1) ] = labels[i]
    # for i, layer in enumerate(fake_data):
    #     fake_data_replays['layer' + str(i+1) ] = fake_data[i]
    # torch.save(replay_data, '1115-replay_data_vermilion_models-2experts.pth')
    torch.save(replay_data, '1121-replay_data_rose_models-2experts.pth')

    # torch.save(layer_replays, 'layer_replays_vermilion_models-2experts.pth')
    # torch.save(input_layer_replays, 'input_layer_replays_new_models-2experts.pth')
    # torch.save(atten_layer_replays, 'atten_layer_replays_new_models-2experts.pth')
    # torch.save(label_layer_replays, 'label_layer_replays_new_models-2experts.pth')
    # torch.save(fake_data_replays, 'fake_data_replays_new_models-2experts.pth')
    test = load_layer_data('1121-replay_data_rose_models-2experts.pth')
    cc = 0
    just_1 = 0
    for k in range(len(test)):
        if len(test[k]):
            # print(k,len(test[k]))
            cc+=1
            if len(test[k]) == 1:
                just_1 += 1
            # print(len(test[k]))
        if len(test[k]) and len(test[4095-k]):
            print(k)
    print(cc,just_1)
    # print(test[1])
    # print(len(test[1][2]))
    # print(test[1][2].shape)

    




    # accelerator.print(f'Number of Samples batches: {len(all_layer_outputs[0])}')
    
    # calculate pca




if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/new_model.json')
    old_dataset = MixedData_1121(config=config)
    

    new_dataset = ACLForLM_1103(config)
    
    # model = base_models.vermilion_model(config=config)
    model = base_models.rose_model(config=config)
    router = base_models.BertWithSavers(config=config)
    # model = base_models.BertWithDecoders(config=config)
    # model = nn.DataParallel(model)
    
    
    layer_pca(model=model, old_dataset=old_dataset,new_dataset=new_dataset, router = router)
