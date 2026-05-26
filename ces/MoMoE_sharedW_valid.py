import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import MoMoE_MIXED_LEGAL_REVIEW,MoMoE_LARGE_SPECIFIC_GENERAL,MoMoE_MIXED_LEGAL_PUBMED,MoMoE_MIXED_0128,MoMoE_MIXED_WIKI103_0124,MoMoE_MIXED,MoMoE_WIKI103,MixedData_1211,MixedData_0110_1,MixedData_1211_1,RestaurantForLM_small, MixedData,MixedData_stage1,Mixdata_1103,Mixdata_1115,Wikitxt103ForLM_1103,Wikitxt103ForLM_0102_rose,MoMoE_FEWER_SPECIFIC_GENERAL
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim
from sklearn.decomposition import PCA
import torch
import numpy as np
import random
from sklearn.cluster import KMeans


def get_clusters(data,k):
    data0 = data.mean(1).cpu().detach().numpy()

    kmeans = KMeans(n_clusters=k, random_state=0).fit(data0)

    # 获取聚类标签
    labels = kmeans.labels_

    # 根据聚类结果分离数据
    data_class= []
    print(labels)
    for i in range(k):
        print(data[labels == i,:,:])
        data_class.append(data[labels == i,:,:])


    return data_class


def get_gds_matrices(embeddings, unique_dims = -1):
    # embeddings : [C, N, M, d]: C clusters, each N samples with dim d
    # for each sampled embeddings in embeddings, cal the PCA transform matrix
    PCA_matrices = []
    for embedding in embeddings:
        embedding = embedding[:768].mean(1).cpu().detach().numpy()
        pca = PCA()
        pca.fit(embedding)
        cmps = pca.components_
        proj = cmps # cmps * cmps.T
        PCA_matrices.append(proj)
    proj = sum(PCA_matrices)
    pca = PCA()
    pca.fit(proj) # np.linalg.eig(proj)
    eig_vals = pca.singular_values_[::-1]
    eig_vecs = pca.components_[::-1]
    # unique dims is the number of eigenvalues that are smaller than 1
    if unique_dims == -1:
        unique_dims = sum(eig_vals < 1)
    print(f'unique dims: {unique_dims}')

    # return the transform matrix
    return eig_vecs.copy(), unique_dims

def get_gds(embeddings, unique_dims = -1):
    # embeddings : [C, N, M, d]: C clusters, each N samples with dim d
    # for each sampled embeddings in embeddings, cal the PCA transform matrix
    PCA_matrices = []
    for embedding in embeddings:
        embedding = embedding[:768].mean(1).cpu().detach().numpy()
        pca = PCA()
        pca.fit(embedding)
        cmps = pca.components_
        proj = cmps # cmps * cmps.T
        PCA_matrices.append(proj)
    proj = sum(PCA_matrices)
    pca = PCA()
    pca.fit(proj) # np.linalg.eig(proj)
    eig_vecs = pca.components_

    return torch.tensor(eig_vecs)



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


def validate(model, val_loader, accelerator, device, centers,model0,device2,PRO_VERS):
    losses = []
    for i, batch in enumerate(val_loader):  
        batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
        batch = {key: tensor.to(device) for key, tensor in batch.items()}

        _,_,_,_,_,_,inputs = model0(**batch0)
        inputs = [i0.to(device) for i0 in inputs]
        with torch.no_grad():
            
            loss, _, _, _, _,_,_,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], centers,inputs,PRO_VERS)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity

# def validate(model, val_loader, accelerator, device, centers,model0,device2):
#     losses = []
#     for i, batch in enumerate(val_loader):  
#         batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
#         batch = {key: tensor.to(device) for key, tensor in batch.items()}

#         _,_,_,_,_,_,inputs = model0(**batch0)
#         inputs = [i0.to(device) for i0 in inputs]
#         with torch.no_grad():
            
#             loss, _, _, _, _,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], centers,inputs)
#         losses.append(accelerator.gather(loss.repeat(len(batch))))
    
#     losses = torch.cat(losses)[:len(val_loader.dataset)]
#     perplexity = torch.mean(losses)
    
#     return perplexity

# def validate(model, val_loader, accelerator, device, centers):
#     losses = []
#     for i, batch in enumerate(val_loader):  

#         batch = {key: tensor.to(device) for key, tensor in batch.items()}
#         with torch.no_grad():
            
#             loss, _, _, _, _,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], centers)
#         losses.append(accelerator.gather(loss.repeat(len(batch))))
    
#     losses = torch.cat(losses)[:len(val_loader.dataset)]
#     perplexity = torch.mean(losses)
    
#     return perplexity


# def validate(model, val_loader, accelerator, device):
#     losses = []
#     model.deval = 1
#     for i, batch in enumerate(val_loader):  

#         batch = {key: tensor.to(device) for key, tensor in batch.items()}
#         with torch.no_grad():
            
#             loss, _, _, _, _,_,_,_ = model(**batch)
#         losses.append(accelerator.gather(loss.repeat(len(batch))))
    
#     losses = torch.cat(losses)[:len(val_loader.dataset)]
#     perplexity = torch.mean(losses)
#     model.deval = 0
#     return perplexity

def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    return layer_data

def load_layer_pro_vec(path):
    layer_data = torch.load(path, map_location='cuda')

    return layer_data

def get_gradient_norms(model):
    """Utility function to get gradient norms of a model."""
    return [param.grad.norm().item() for param in model.parameters() if param.grad is not None]
def get_routes_ids(expert_ids,config):
    IDS = []
    for i in range(expert_ids[0].shape[0]):
        c = 0
        for j in range(len(expert_ids)):
            c+=expert_ids[j][i]*2**j
        IDS.append(c)
    return IDS

def train(model, num_epochs, dataset, device, ahead_dataset):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    
    freze_emb = 0 
    AngeL = 0
    sparsity_rate = 0
    SPAR = 1
    orth_rate = 0
    ORTH = 1
    same_in_param = 0

    with_bert_param = 1


    model0 = base_models.BertForMLM_toshow(config=config0)
    model01 = torch.load('MODELS_0315/0318-MoMoE_full_dimension-3072ffns-last4-150+350W*1-2e-4.pth')
    model01 = torch.load('0226-bert-batch64_seq128-MIXED_LEGAL_PUBMED-pretrain-lr2e-4-for-routing2.pth')

    model0.load_state_dict(model01.state_dict())


    # cluster_centers = load_layer_data('0226-layer_centers-2t-MIXED_LEGAL_PUBMED2.pth')
    cluster_centers = load_layer_data('0226-layer_centers-2t-MIXED_LEGAL_PUBMED2.pth')


    train_loader=dataset.val_loader_f
    # ahead_train_loader = ahead_dataset.train_loader
    # ahead_val_loader = ahead_dataset.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    # writer = SummaryWriter('tensorboard_0119/0119_MoMoE_batch64_seq128_4t16e_WIKI103_WP10_LAR3')
    writer = SummaryWriter('tensorboard_0315/0315-MoMoE_full_dimension-3072ffns-150+350W*1-1.5e-4')
    # writer = SummaryWriter('tensorboard_0311/0311_MoMoE_sharedW_64*128_3t4e_LARGE_SPECIFIC_GENERAL-lr3e-4')

    
    # writer = SummaryWriter('0118_MoMo_CAU_v1_batch32_seq256-show')


    
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    # model, optimizer, lr_scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    model, optimizer, lr_scheduler, train_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader)
    model.to(device)

    device2 = torch.device("cuda:3")
    model0.to(device2)
    # router.to(device)
    if freze_emb:
        for para in model.embeddings.parameters():
            para.requires_grad = False
    
    ####for unique and common dimension#####
    step0 = 0
    VECTORS_PRO = []
    model0.eval()

    PRO_VECS = []
    for l in range(config.num_hidden_layers):
        eig_vecs0 = torch.load('0229-layer%d_pro_vec-2t-MIXED_LEGAL_PUBMED.pth'%l, map_location='cuda')
        PRO_VECS.append(eig_vecs0)
    # print("2222",eig_vecs0.shape)


        
    steps = 0
    losses = []

    for epoch in range(1):
        
        model.eval()
        
        
        
        # routes_history = [0 for i in range(4**12)]
        for i, batch in enumerate(train_loader):
            # print(i)
            steps+=1
            batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
            batch = {key: tensor.to(device) for key, tensor in batch.items()}

            
            _,_,_,_,_,_,inputs = model0(**batch0)

            inputs = [i0.to(device) for i0 in inputs]

            loss, _, _, _, ids,_,_,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers,inputs,PRO_VECS)

            # print(layers_o[7].shape)

            losses.append(accelerator.gather(loss.repeat(config.batch_size)))

    
            if steps%100 == 0:
                print(f"steps:{steps}")
                print(ids)
        
                loss_val = torch.mean(torch.cat(losses)[:len(losses)])

                if accelerator.is_local_main_process:
                    writer.add_scalar('perplexity_valid_step', loss_val, steps)

                losses = []


    


    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/MoMoE.json')
    config0 = BertConfig.from_json_file('config/bert.json')

    # config = BertConfig.from_json_file('config/AngeL_rose2.json')

    # dataset = RestaurantForLM_small(config=config)
    # dataset = MoMoE_MIXED_WIKI103_0124(config = config)
    # ahead_dataset = MoMoE_MIXED_WIKI103_0124(config = config)
    dataset = MoMoE_MIXED_LEGAL_PUBMED(config = config)
    ahead_dataset = MoMoE_MIXED_LEGAL_PUBMED(config = config)
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    model = base_models.MoMoE_0126(config=config)
    # model = base_models.MoMoE_0306_narrowneck(config=config)
    # model = base_models.MoMoE_0313(config=config)
    # model = base_models.MoMoE_0315(config=config)

    
    # model = base_models.MoMoE_salary1062(config=config)
    # model = base_models.MoMoE_uniqueatt_commonffn(config=config)

    # model = base_models.MoMoE_CAU(config=config)
    # model = base_models.MoMoE_prenorm(config=config)




    # router = base_models.BertWithSavers(config=config)
    # router.to(device)
    # model.to(device)
    # model = nn.DataParallel(model)

    
    train(model=model, num_epochs=1, dataset=dataset, device=device, ahead_dataset = ahead_dataset)