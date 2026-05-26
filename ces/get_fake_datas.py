import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import RestaurantForLM_small, ACLForLM_small, RestaurantForLM,Mixdata_1103, ACLForLM_1103
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

import random
from transformers.models.bert.modeling_bert import BertOnlyMLMHead,BertPooler

def get_fake_data(input, c,r,k):



    X = input.mean(axis=1)
    X = X.cpu().numpy()

    kmeans = KMeans(n_clusters=c, n_init= 'auto')
    
    # dbscan = DBSCAN(eps=50, min_samples=3)
    # ypred = dbscan.fit_predict(output)
    # print(f'DBSCAN labels: {max(ypred)}')
    # Fit the model to your data
    kmeans.fit(X)

    # Get cluster assignments for each data point
    # labels = kmeans.labels_
    # print(f'cluster_num {labels}')
    # Get the coordinates of the cluster centers
    cluster_centers = kmeans.cluster_centers_
    # print(f'cluster_centers {cluster_centers}')


    cluster_assignments = kmeans.labels_

    outputs = [[]for i in range(c)]
    outputs_ids = [[]for i in range(c)]
    outputs_ids_return = []
    for cl in  range(len(cluster_assignments)):
        # outputs[cluster_assignments[cl]].append(input[cl])
        outputs_ids[cluster_assignments[cl]].append(cl)
    # print(outputs)
    # print(outputs_ids)
    # for j in  range(len(outputs)):
    #     # outputs[cluster_assignments[cl]]=torch.tensor(outputs[cluster_assignments[cl]])
    #     outputs[j]=torch.cat(outputs[j], dim=0)

    for l in range(len(outputs_ids)):
        random.shuffle(outputs_ids[l])
        outputs_ids_return.append(outputs_ids[l][0:r*k])
    
    return outputs_ids_return 


def get_fake_sample(output, c, r, k):

    data = output.mean(1).cpu().numpy()

    # 定义要聚类的簇数
    num_clusters = c

    # 使用sklearn的KMeans进行聚类
    kmeans = KMeans(n_clusters=num_clusters, n_init= 'auto')
    cluster_assignments = kmeans.fit_predict(data)

    # 将数据和聚类结果转换为PyTorch张量
    data_tensor = torch.tensor(data, dtype=torch.float)
    cluster_assignments_tensor = torch.tensor(cluster_assignments)

    # 找到每个簇中离中心最近的k个数据点
    num_to_save = r*k
    nearest_points = []
    nearest_ids = []
    for i in range(num_clusters):
        cluster_data = data_tensor[cluster_assignments_tensor == i]
        cluster_centers = kmeans.cluster_centers_[i]
        cluster_distances = torch.norm(cluster_data - cluster_centers, dim=1)
        nearest_indices = torch.argsort(cluster_distances)[:num_to_save]
        nearest_points.append(cluster_data[nearest_indices])
        nearest_ids.append(nearest_indices.tolist())
        print(len(nearest_indices))
    
    return nearest_ids


    
def output2fakedata(output):
    n = output.shape[0]
    d = output.shape[1]
    fake_input = torch.zeros(n,d)
    #softmax_2 = nn.Softmax(dim=2)
    #self.output = softmax_2(self.output)

    index_id = torch.argmax(output, 2)
    for i in range(n):
        for j in range(d):
            fake_input[i][j] = index_id[i][j]
    
    return fake_input

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


def get_cluster_centers(input,k =5):
    X = input.mean(axis=1)
    X = X.cpu().numpy()
    kmeans = KMeans(n_clusters=k, n_init= 'auto')
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_
    
    return cluster_centers




def layer_pca(model, dataset):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    
    num_updates = 70 * len(train_loader)
    model0 = torch.load('bert-1103-stage0.pth')
    model.bert.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    for i in range(config.num_hidden_layers):
        model.bert.layers.layers[i].load_state_dict(model0.bert.encoders.layers[i].state_dict())
    model.head.load_state_dict(model0.head.state_dict())
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()
    
    # load model checkpoint
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
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
    test_outputs = [[] for i in range(12)]
    test_ffn = [[] for i in range(12)]
    fake_datas = [[] for i in range(12)]
    ids = []
    # pooler = BertPooler(config).to('cuda')
    memory = MemoryFromDecoder()
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i %50 ==0 and i < 1000:          
                print(f"######{i}")                
                _, scores, layer_outputs, ffn_outputs = model(**batch)


                

                pooled_scores = output2fakedata(scores)
                
                # print(pooled_scores.shape)
                # scores.to('cpu')
                for j, layer_output in enumerate(layer_outputs): 
                    # print(layer_output.shape) 
                    # layer_output = layer_output.view(config.batch_size,-1)
                    test_outputs[j].append(layer_output)
                    fake_datas[j].append(pooled_scores)
                    test_ffn[j].append(ffn_outputs[j])
                    # test_output = get_fake_data(layer_output,3,4,3)
            if i ==1000:
                break
    for j in range(len(test_outputs)):
        test_outputs[j] = torch.cat(test_outputs[j],dim = 0)
    for j in range(len(test_ffn)):
        test_ffn[j] = torch.cat(test_ffn[j],dim = 0)
    for j in range(len(fake_datas)):
        fake_datas[j] = torch.cat(fake_datas[j],dim = 0)
    for j in range(len(test_outputs)):
        ids.append(get_fake_data(test_outputs[j],2,8,8))
        # ids.append(get_fake_sample(test_outputs[j],2,8,8))
        # ids.append(get_fake_sample(test_ffn[j],2,8,8))
    
    # print(len(ids[-1]))
    # print(len(ids[-1][1]))
    # print(ids[-1][1][0])
    # print(fake_datas[10].shape)

    layer_fake_dtats = {}
    layer_fake_dtats_ids = {}

    for i, layer in enumerate(fake_datas):
        # print(ids[i])
        # print(fake_datas[i].shape)
        layer_fake_dtats[str(i)] =fake_datas[i][np.concatenate(ids[i])]
    for i, layer in enumerate(ids):
        layer_fake_dtats_ids[str(i)] =np.concatenate(ids[i])
    torch.save(layer_fake_dtats, 'layer_fake_dtats-shuffle-2-128.pth')
    torch.save(layer_fake_dtats_ids, 'layer_fake_dtats_ids-shuffle-2-128.pth')

    
    T_layer_fake_dtats = load_layer_data('layer_fake_dtats-shuffle-2-128.pth')
    T_layer_fake_dtats_ids = load_layer_data('layer_fake_dtats_ids-shuffle-2-128.pth')
    # old_T_layer_fake_dtats_ids = load_layer_data('layer_fake_dtats_ids.pth')

    print(T_layer_fake_dtats[-1].shape)
    print(len(T_layer_fake_dtats_ids))
    print(len(T_layer_fake_dtats_ids[-1]))
    # print(len(T_layer_fake_dtats_ids[-1][0]))

    print(T_layer_fake_dtats_ids[-1][0])
    for j in range(12):
        print(T_layer_fake_dtats_ids[j][0])
    # cc = 0
    # for k in range(test_outputs[0].shape[0]):
    #     ktt = 0
    #     for jj in range(len(T_layer_fake_dtats_ids)):
    #         # print(T_layer_fake_dtats_ids[jj])
    #         if k in T_layer_fake_dtats_ids[jj]:
    #             ktt +=1
    #     if ktt >=6:
    #         cc+=1
    #         print(k)
    # print(f'cc = {cc}')

    


    









        

    





    # accelerator.print(f'Number of Samples batches: {len(all_layer_outputs[0])}')
    
    # calculate pca




if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/bert.json')
    dataset = Mixdata_1103(config=config)
    
    model = base_models.BertWithSavers(config=config)
    # model = base_models.BertWithDecoders(config=config)
    # model = nn.DataParallel(model)
    
    layer_pca(model=model, dataset=dataset)
