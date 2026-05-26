import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import MoMoE_longtailed,Wikipedia,MoMoE_MIXED_LEGAL_REVIEW,MoMoE_FEWER_SPECIFIC_GENERAL,MoMoE_MIXED_LEGAL_PUBMED,MoMoE_MIXED_WARMUP,MoMoE_WIKI103_WARMUP,MixedData_0110_0,MixedData_1211_0,RestaurantForLM_small, ACLForLM_small, MixedData, MixedData_1121,Wikitxt103ForLM_0102_warmup,LegalForLM,PubMedForLM
from accelerate import Accelerator,load_checkpoint_and_dispatch
from transformer.Transformer_MOE import BertModel,BertModel_every2layers
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertOnlyNSPHead
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
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from Dataset_new import GLUE,GAD,Overruling,GAD_single,medical_abstract,multi_domains,Casehold,EUADR
class BertForMLM_every2layers(nn.Module):
    def __init__(self, config):
        super(BertForMLM_every2layers, self).__init__()
        self.config = config
        self.bert = BertModel_every2layers(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask, labels):
        output,att_sself,O,_ = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores,att_sself,O

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

def show_clusters(input,l,name):
    # lora_mat_1 = torch.load('0514-lora_mat.pth')
    X = input.mean(1).cpu().numpy()
    X_O = X
    pca = PCA()
    pca.fit(X)
    c= 1
    # 获取正交基（主成分）
    components = pca.components_
    mean = pca.mean_

    # 按照信息量排序（即按照解释的方差大小排序）
    variance_explained = pca.explained_variance_ratio_
    sorted_indices = np.argsort(variance_explained)[::-1]
    sorted_components = components[sorted_indices]

    # 每10个正交基一组，转换数据
    transformed_data_groups = []
    for i in range(0, 5, c):
        # 选择当前组的10个正交基
        group_components = sorted_components[i:i+c]
        # 转换数据
        pca_group = PCA(n_components=i+c)
        pca_group.components_ = group_components
        pca_group.mean_ = mean
        transformed_data = pca_group.transform(X_O)
        # transformed_data_groups.append(transformed_data)
        # X = torch.matmul(input.mean(1), lora_mat_1.to(device)).cpu().numpy()
        XO = umap.UMAP(random_state=42).fit_transform(transformed_data)
    
        plt.figure(l+100*i)
        # plt.scatter(X[:,0],X[:,1],label = name)
        plt.scatter(XO[:-240,0],XO[:-240,1],color = "b")
        plt.scatter(XO[-240:-120,0],XO[-240:-120,1],color = "g")
        plt.scatter(XO[-120:,0],XO[-120:,1],color = "y")
        # plt.legend()
        plt.xlim(-10,20)
        plt.ylim(-10,20)

        # plt.scatter(original_centers0[:,0],original_centers0[:,1],color = "r")
        plt.savefig("0630-bert/0630-cluster-bert-l%d-%d-components-only%d.png"%(l,name,i))
    for i in range(0, 5, c):
        # 选择当前组的10个正交基
        group_components = sorted_components[:i+c]
        # 转换数据
        pca_group = PCA(n_components=i+c)
        pca_group.components_ = group_components
        pca_group.mean_ = mean
        transformed_data = pca_group.transform(X_O)
        # transformed_data_groups.append(transformed_data)
        # X = torch.matmul(input.mean(1), lora_mat_1.to(device)).cpu().numpy()
        XO = umap.UMAP(random_state=42).fit_transform(transformed_data)
    
        plt.figure(l+100000*i)
        # plt.scatter(X[:,0],X[:,1],label = name)
        plt.scatter(XO[:-240,0],XO[:-240,1],color = "b")
        plt.scatter(XO[-240:-120,0],XO[-240:-120,1],color = "g")
        plt.scatter(XO[-120:,0],XO[-120:,1],color = "y")
        # plt.legend()
        plt.xlim(-10,20)
        plt.ylim(-10,20)

        # plt.scatter(original_centers0[:,0],original_centers0[:,1],color = "r")
        plt.savefig("0630-bert/0630-cluster-bert-l%d-%d-components-%d-%d.png"%(l,name,0,i+1))
    # X = umap.UMAP(random_state=42).fit_transform(X)
    
    # plt.figure(l+100000)
    # # plt.scatter(X[:,0],X[:,1],label = name)
    # plt.scatter(X[:-240,0],X[:-240,1],color = "b")
    # plt.scatter(X[-240:-120,0],X[-240:-120,1],color = "g")
    # plt.scatter(X[-120:,0],X[-120:,1],color = "y")
    # # plt.legend()

    # # plt.scatter(original_centers0[:,0],original_centers0[:,1],color = "r")
    # plt.savefig("0630-bert/0630-cluster-bert-l%d-%d.png"%(l,name))

def show_clusters_DBSCAN(input,l,name):
    lora_mat_1 = torch.load('0514-lora_mat.pth')
    X= torch.matmul(input.mean(1), lora_mat_1.to(device)).cpu().numpy()
    # # XDIF,_ =get_topk_variences(input,100)
    # X1,index1 = get_dopk_variences(input[-240:-120],100)
    # X2,index2 = get_dopk_variences(input[-120:],100)
    # X3,index3 = get_dopk_variences(input[:-240],100)
    # merged_index = torch.cat((index1, index2, index3))
    
    


    # # 获取全集
    # full_set = torch.unique(merged_index)

    # print(len(full_set))

    # X = input.mean(1)[:, full_set].cpu().numpy()
    # XSIM = input.view(input.shape[0],-1)[:, full_set]
    # X= torch.cat((XDIF,XSIM),dim = 1).cpu().numpy()
    # X= XDIF.cpu().numpy()



    # X = input.mean(1).cpu().numpy()
    # X = input.cpu().numpy()
    

    dbscan = KMeans(3)
    clusters = dbscan.fit_predict(X)
    print(clusters)
    X = umap.UMAP(random_state=42).fit_transform(X)
    
    plt.figure(l)
    plt.scatter(X[:, 0], X[:, 1], c=clusters,cmap='viridis')
    # plt.scatter(X[:, 0], X[:, 1], label = name)
    # plt.scatter(X[:,0],X[:,1],label = name)
    # plt.scatter(X[:-240,0],X[:-240,1],color = "b")
    # plt.scatter(X[-240:-120,0],X[-240:-120,1],color = "g")
    # plt.scatter(X[-120:,0],X[-120:,1],color = "y")
    # plt.legend()
    # plt.xlim(-5,15)
    # plt.ylim(-5,15)
    # plt.tick_params(axis='x', labelsize=1)
    # # plt.tick_params(axis='y', labelsize=1)
    # plt.locator_params(axis='x', nbins=11)
    # plt.locator_params(axis='y', nbins=11)

    # plt.scatter(original_centers0[:,0],original_centers0[:,1],color = "r")
    plt.savefig("0517-cluster-kmeans-bert-l-%d-steps.png"%l)
def validate(model, val_loader, accelerator, device):
    losses = []
    for i, batch in enumerate(val_loader):  
        # batch = {key: tensor.to(device) for key, tensor in batch.items()}      
        with torch.no_grad():
            loss, _ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity
def show_momoe_clusters(input,l,ws):
    X = input.mean(1).cpu().numpy()
    ws = ws.cpu().numpy()

    X = umap.UMAP(random_state=42).fit_transform(X)
    
    plt.figure(l+100)
    plt.scatter(X[:-320,0],X[:-320,1],color = "b",label = ws[:-320])
    plt.scatter(X[-320:-160,0],X[-320:-160,1],color = "g",label = ws[-320:-160])
    plt.scatter(X[-160:,0],X[-160:,1],color = "y",label = ws[-160:])
    plt.legend()
    # plt.scatter(original_centers0[:,0],original_centers0[:,1],color = "r")
    plt.savefig("0506-momoe-cluster-l%d.png"%l)


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

def get_topk_variences(hidden_states,k):
    variances = torch.var(hidden_states.view(hidden_states.shape[0],-1), dim=0)

    _, top_indices = torch.topk(variances, k)
    top_features = hidden_states.view(hidden_states.shape[0],-1)[:, top_indices]
    return top_features,top_indices
def get_dopk_variences(hidden_states,k):
    variances = torch.var(hidden_states.mean(1), dim=0)

    _, top_indices = torch.topk(variances, k,largest=False)
    top_features = hidden_states.mean(1)[:, top_indices]
    return top_features,top_indices

def layer_pca(model, dataset,name,LEGAL):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader1, train_loader2,train_loader3 = dataset.train_loader1, dataset.train_loader2,dataset.train_loader3
    val_loader1 = dataset.val_loader_domain1
    val_loader2 = dataset.val_loader_domain2
    num_updates = 50 * len(train_loader)
    # model01 = base_models.BertForMLM(config=config)
    # model0 = torch.load('0311-BERT-FOR-CLUSTER-MIXED_LEGAL_REVIEW.pth')
    # load_checkpoint_and_dispatch(model, '/home/jxzhou/PLM_PER/1227/MODELS_0502/MOE2-768ffns-wikipedia-large-1.5e-4',device_map={"": device}) 
    load_checkpoint_and_dispatch(model, '/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_REBUTALL/BERT_WIKI%d'%LEGAL,device_map={"": device}) 
    
    # model.load_state_dict(model01.state_dict())

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()
    # model.bert.embeddings.load_state_dict(model01.bert.embeddings.state_dict())
    # for i in range(config.num_hidden_layers):
    #     model.bert.layers.layers[i].load_state_dict(model01.bert.encoders.layers[i].state_dict())
    # model.head.load_state_dict(model01.head.state_dict())
    # load model checkpoint
    # model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)

    model, optimizer, lr_scheduler, train_loader, val_loader,train_loader1, train_loader2,train_loader3,val_loader1,val_loader2= accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader,train_loader1, train_loader2,train_loader3,val_loader1,val_loader2)
    # accelerator.load_state(load_path)
    print(validate(model, val_loader1, accelerator, device))
    print(validate(model, val_loader2, accelerator, device))

    # run once
    model.eval()

    out_for_cluster = [[] for i in range(12)]
    momoe_out_for_cluster = [[] for i in range(12)]
    ROUTES = [[] for i in range(12)]

    if name == 'wikipedia':
        k = 1500
    else: k=1500


    
    with torch.no_grad():
        for i, batch in enumerate(train_loader3):
            if i>=100:
                break
            if i %40 == 0:          
                print(f"######{i}")                
                _,outputs = model.bert(batch['input_ids'],batch['attention_mask'])
                # _,_,outputs,_ = model.bert(batch['input_ids'],batch['attention_mask'])

                for j, layer_output in enumerate(outputs):  
                    out_for_cluster[j].append(layer_output.detach())
                    
                    # out_for_cluster[j].append(get_dopk_variences(layer_output.detach().mean(1),100))
                    # if j == 10 or j == 8:
                    #     momoe_out_for_cluster[j].append(layer_output)
                    #     ROUTES[j].append(routes[j])

        for i, batch in enumerate(train_loader1):
            if i>=100:
                break
            if i %40 == 0:          
                print(f"######{i}")                
                _,outputs = model.bert(batch['input_ids'],batch['attention_mask'])

                for j, layer_output in enumerate(outputs):  
                    
                    out_for_cluster[j].append(layer_output.detach())
                    # out_for_cluster[j].append(get_dopk_variences(layer_output.detach().mean(1),100))



        for i, batch in enumerate(train_loader2):
            if i>=100:
                break
            if i %40 == 0:          
                print(f"######{i}")                
                _,outputs = model.bert(batch['input_ids'],batch['attention_mask'])

                for j, layer_output in enumerate(outputs):  
                    
                    out_for_cluster[j].append(layer_output.detach())
                    # out_for_cluster[j].append(get_dopk_variences(layer_output.detach().mean(1),100))

    for j in range(len(out_for_cluster)):
        out_for_cluster[j] = torch.cat(out_for_cluster[j], dim =0 ).detach()
    # for j in range(12):
    #     if len(momoe_out_for_cluster[j]):
    #         print(ROUTES[j])
    #         momoe_out_for_cluster[j] = torch.cat(momoe_out_for_cluster[j], dim =0 ).detach()
    #         ROUTES[j] = torch.cat(ROUTES[j], dim =0 ).detach()
    for j in range(len(out_for_cluster)):
        show_clusters(out_for_cluster[j],j,LEGAL)
        # show_clusters(out_for_cluster[j],j,name)

        # if len(momoe_out_for_cluster[j]):
            
        #     show_momoe_clusters(momoe_out_for_cluster[j],j,ROUTES[j])




if __name__ == "__main__":
    set_seed(45)
    
    # config = BertConfig.from_json_file('config/bert.json')
    config = BertConfig.from_json_file('config/bert.json')
    device = torch.device("cuda")

    # model = BertForMLM_every2layers(config=config)
    model = base_models.BertForMLM_showing(config=config)

    # dataset = MoMoE_longtailed(config=config)
    # for i in range(7000,80000,14000):
    #     layer_pca(model=model, dataset=dataset,name = i)
    datasets = {}
    # datasets['casehold'] = Casehold(config)
    # datasets['overruling'] = Overruling(config)

    # datasets['sst2'] = GLUE(config)

    # datasets['gad'] = GAD(config)
    # datasets['euadr'] = EUADR(config)
    # new_dataset = MoMoE_MIXED_LEGAL_REVIEW(config)
    torch.cuda.set_device(6)
    device = torch.device("cuda:6")

    datasets['mixed'] = MoMoE_longtailed(config,KKK = 100000)
    # datasets['legal'] = LegalForLM(config)
    # datasets['pubmed'] = PubMedForLM(config)
    
    LEGAL = 1500000

    # model = base_models.BertWithDecoders(config=config)
    # model = nn.DataParallel(model)
    # 
    # load_path = "./output-formal-1X"
    for i in datasets:
        layer_pca(model=model, dataset=datasets[i],name = i,LEGAL= LEGAL)
