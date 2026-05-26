import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import Wikipedia_small,Wikipedia,Wikipedia_Wids,MoMoE_longtailed,MoMoE_MIXED_LEGAL_REVIEW,MoMoE_LARGE_SPECIFIC_GENERAL,MoMoE_MIXED_LEGAL_PUBMED,MoMoE_MIXED_0128,MoMoE_MIXED_WIKI103_0124,MoMoE_MIXED,MoMoE_WIKI103,MixedData_1211,MixedData_0110_1,MixedData_1211_1,RestaurantForLM_small, MixedData,MixedData_stage1,Mixdata_1103,Mixdata_1115,Wikitxt103ForLM_1103,Wikitxt103ForLM_0102_rose,MoMoE_FEWER_SPECIFIC_GENERAL
from accelerate import Accelerator,load_checkpoint_and_dispatch
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim
from sklearn.decomposition import PCA
import torch
import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import util
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
def pca_projection_with_torch(input,projects,k0):
    input = input.cpu().numpy()
    matrix = projects.cpu().numpy()

    pca = PCA(n_components=k0,whiten=True)  # 假设你想要降维到2维
    pca.fit(matrix)

    # 假设你有一个单独的张量
    if not np.allclose(input, np.zeros_like(input)):
        tensor = input / np.linalg.norm(input)
    else:
        tensor = input

    # 使用PCA的转换基来转换张量
    transformed_tensor = torch.tensor(pca.transform(tensor))
    
    norm_t = torch.sum(transformed_tensor)/k0
    # print(norm_t)
    return transformed_tensor, norm_t

def pca_similarity_with_torch(input,projects,k0):
    input = input.cpu().numpy()
    matrix = projects.cpu().numpy()

    pca = PCA(n_components=k0,whiten=True)  # 假设你想要降维到2维
    pca.fit(matrix)
    # explained_variance_ratio = pca.explained_variance_ratio_
    # print(explained_variance_ratio[0],explained_variance_ratio[-1])
    transformed_basis = pca.components_

    # 假设你有一个单独的张量
    if not np.allclose(input, np.zeros_like(input)):
        tensor = input / np.linalg.norm(input)
    else:
        tensor = input
    # 使用PCA的转换基来转换张量
    transformed_tensor = torch.tensor(pca.transform(tensor))

    cosine_similarities = cosine_similarity(tensor, transformed_basis)

    cosine_similarities = torch.tensor(np.mean(cosine_similarities))
    
    return transformed_tensor, cosine_similarities


def get_ntk(model: base_models.MoMoE_0126, loader, accelerator: Accelerator,cluster_centers,PRO_VECS,model0,device2):
    
    module = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    config = module.config

    grads = {}
    NTKs_norm = {}
    
    head_grads = [[[] for _ in range(config.num_attention_heads)] for _ in range(config.num_hidden_layers)]
    head_NTKs_norm = [[0 for _ in range(config.num_attention_heads)] for _ in range(config.num_hidden_layers)]
    expert_NTKs_norm = [[0 for _ in range(config.num_experts)] for _ in range(config.num_hidden_layers)]

    for i, batch in enumerate(loader):
        if i >= 1: break
        # batch = {k: v[:1, :] for k, v in batch.items()}
        batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
        
            
        _,_,_,_,_,_,inputs = model0(**batch0)
        inputs = [i0.to(device) for i0 in inputs]




        _, logits, _, _, _,_,_,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers,inputs,PRO_VECS)
        # logits = logits.view(-1, config.vocab_size)

        for i in range(len(logits)):
            # b, s, v = logits.shape
            # param_grads = torch.autograd.grad(logits[i:i+1], model.parameters(), grad_outputs=torch.ones_like(logits[i:i+1]), create_graph=True)
            logits[i:i+1].backward(torch.ones_like(logits[i:i+1]))
            print(i)
            # accelerator.backward(logits[i:i+1], gradient=torch.ones_like(logits[i:i+1]))

            for name, param in module.named_parameters():
                # print(name)
                if not 'layers' in name: continue # ignore embedding layers
                if param.requires_grad:
                    grad = grads.get(name, [])
                    grad.append(accelerator.gather(param.grad.detach().reshape(-1).unsqueeze(0).clone()))
                    grads[name] = grad
                if 'heads' in name and 'attentions.2' in name:
                    grad = param.grad.detach().flatten()
                    splited = name.split('.')
                    l = int(splited[splited.index('layers') + 1])
                    h = int(splited[splited.index('heads') + 1])
                    # print(name)
                    head_grads[l][h].append(accelerator.gather(grad.unsqueeze(0).clone()))
                    
            
            model.zero_grad()
            torch.cuda.empty_cache()

    for name, grad in grads.items():
        J_layer = torch.concat(grad)

        J_layer_norm = J_layer.T / torch.norm(J_layer.T, dim=0)
        NTK_norm = J_layer_norm.T @ J_layer_norm
        NTKs_norm[name] = NTK_norm

        if 'layers' in name:
            splited = name.split('.')
            l = int(splited[splited.index('layers') + 1])
            if 'experts' in name:
                e = int(splited[splited.index('experts') + 1])
                expert_NTKs_norm[l][e] += NTKs_norm[name]
            
    for l, layer in enumerate(head_grads):
        for h, head in enumerate(layer):
            J_head = torch.concat(head)
            # print(J_head.shape)
            J_head_norm = J_head.T / torch.norm(J_head.T, dim=0)
            # print(J_head_norm.shape)
            # J_head_norm = J_head.T 

            head_NTK_norm = J_head_norm.T @ J_head_norm


            head_NTKs_norm[l][h] += head_NTK_norm.T
        
        head_NTKs_norm[l] = torch.stack(head_NTKs_norm[l])
        # print(head_NTKs_norm[l].shape)
        expert_NTKs_norm[l] = torch.stack(expert_NTKs_norm[l])
    
    head_NTKs_norm = torch.stack(head_NTKs_norm).cpu() # l, h, n, n
    expert_NTKs_norm = torch.stack(expert_NTKs_norm).cpu() # l, e, n, n

    head_lmax_norm = torch.stack([
        torch.stack([util.lmax(head_NTKs_norm[l][h]) 
        for h in range(config.num_attention_heads)])
        for l in range(config.num_hidden_layers)
    ])

    expert_lmax_norm = torch.stack([
        torch.stack([util.lmax(expert_NTKs_norm[l][e]) 
        for e in range(config.num_experts)])
        for l in range(config.num_hidden_layers)
    ])

    return head_lmax_norm, expert_lmax_norm

def get_ntk_by_layer(model,loader,cluster_centers,PRO_VECS,model0,device2,steps,doamin,longtailed_data,o):

    L , W, H = 12,3,8
    NTKs = [[[[] for k in range(H)]for j in range(W)]for i in range(L)]
    longtailed_NTKs = [[[[] for k in range(H)]for j in range(W)]for i in range(L)]
    J_GRADS = [[[[] for k in range(H)]for j in range(W)]for i in range(L)]

    NTK = torch.zeros(L,W,H)
    longtailed_NTK = torch.zeros(L,W,H)
    longtailed_sim = torch.zeros(L,W,H)

    params_that_need_grad = []
    for param in model.parameters():
        if param.requires_grad:
            params_that_need_grad.append(param.requires_grad)
            param.requires_grad = False # first set all gradients to not calculate, time saver
        else:
            params_that_need_grad.append(param.requires_grad)
    
    for index, (name, param) in enumerate(model.named_parameters()):
        if not params_that_need_grad[index]: #if it didnt need a grad, we can skip it.
            continue
        
        if 'heads' in name and 'attentions' in name:
            param.requires_grad = True #we only care about this tensors gradients in the loop
            this_grad = []
            longtailed_data_this_grad = []
            splited = name.split('.')
            l = int(splited[splited.index('layers') + 1])
            t = int(splited[splited.index('attentions') + 1])
            h = int(splited[splited.index('heads') + 1])
            print(l,t,h)
            if l >= L:break
            print(name)

            for i, batch in enumerate(loader):
                if i >= 1: break
                batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
        
            
                _,_,_,_,_,_,inputs = model0(**batch0)
                inputs = [i0.to(device) for i0 in inputs]
                


                _, logits, _, _, _,_,_,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers,inputs,PRO_VECS)

                for i in range(len(logits)):
                    # print(i)
                    logits[i:i+1].backward(torch.ones_like(logits[i:i+1]), create_graph=True)
                    this_grad.append(param.grad.detach().reshape(-1).clone())
                    model.zero_grad()
                    torch.cuda.empty_cache()

                batch0 = {key: tensor.to(device2) for key, tensor in longtailed_data.items()}
        
                _,_,_,_,_,_,inputs = model0(**batch0)
                inputs = [i0.to(device) for i0 in inputs]
                
                _, logits, _, _, _,_,_,_,_,_ = model(longtailed_data['input_ids'],longtailed_data['attention_mask'], longtailed_data['labels'], cluster_centers,inputs,PRO_VECS)

                for i in range(len(logits)):
                    # print(i)
                    logits[i:i+1].backward(torch.ones_like(logits[i:i+1]), create_graph=True)
                    longtailed_data_this_grad.append(param.grad.detach().reshape(-1).clone())
                    model.zero_grad()
                    torch.cuda.empty_cache()

            J_layer = torch.stack(this_grad) # [N x P matrix] #this will go against our notation, but I'm not adding
            longtailed_J_layer = torch.stack(longtailed_data_this_grad) # [N x P matrix] #this will go against our notation, but I'm not adding
 
            NTKs[l][t][h].append(J_layer @ J_layer.T )# An extra transpose operation to my code for us to feel better
            J_GRADS[l][t][h].append(J_layer)
            # print(NTKs[l][t][h])
            longtailed_NTKs[l][t][h].append(longtailed_J_layer)
            param.requires_grad = False
            model.zero_grad()
            torch.cuda.empty_cache()

            

     
    #reset the model object to be how we started this function
    for i, param in enumerate(model.parameters()):
        if params_that_need_grad[i]:
            param.requires_grad = True
    
    for l in range(L):
        for w in range(W):
            for h in range(H):
                # if len(NTKs[l][w][h]) == 0:
                #     NTK[l][w][h] = torch.tensor(0)
                
                # print(util.lmax(NTKs[l][w][h][0]))
                NTK[l][w][h] = util.lmax(NTKs[l][w][h][0])
                # print(longtailed_NTKs[l][t][h][0])
                _,longtailed_NTK[l][w][h] = pca_projection_with_torch(longtailed_NTKs[l][t][h][0],J_GRADS[l][w][h][0],4)
                # print(longtailed_NTK[l][w][h])
                _,longtailed_sim[l][w][h] = pca_similarity_with_torch(longtailed_NTKs[l][t][h][0],J_GRADS[l][w][h][0],4)
                # print(longtailed_sim[l][w][h])
            # print(NTK[l][w])
            # print(longtailed_NTK[l][w])
            # print(longtailed_sim[l][w])


    lambdamax_layers = NTK

    longtailed_NTK[longtailed_NTK == float('inf')] = 0
    longtailed_sim[longtailed_sim == float('inf')] = 0
    print(torch.max(longtailed_NTK[:,0:2,:], dim=1)[0])

    # print(torch.max(lambdamax_layers[:,0:2,:],dim=1)[0].shape)
    plt.figure(o,dpi=120)
    sns.heatmap(data=torch.max(lambdamax_layers[:,0:2,:],dim=1)[0],
                vmin=1e11,
                vmax=1e16,
                cmap=plt.get_cmap('Blues')
        )
    plt.title('LAMBDAmax_IN_HEADS_EVERY_LAYERS') 
    plt.savefig('NTKS-0330/NTK-lambdamax-max-%dsteps-%s.png'%(steps,doamin)) 
    o+=1

    for w in range(W):
        plt.figure(o,dpi=120)
        sns.heatmap(data=lambdamax_layers[:,w,:],
                    vmin=1e11,
                    vmax=1e16,
                    cmap=plt.get_cmap('Blues')
            )
        plt.title('LAMBDAmax_IN_HEADS_EVERY_LAYERS') 
        plt.savefig('NTKS-0330/NTK-lambdamax-%dsteps-%dW-%s.png'%(steps,w,doamin)) 
        o+=1
    plt.figure(o,dpi=120)
    sns.heatmap(data=torch.max(longtailed_NTK[:,0:2,:], dim=1)[0],
                vmin=-1,
                vmax=1,
                cmap=plt.get_cmap('Blues')
        )
    plt.title('PROJECTIONS_IN_HEADS_EVERY_LAYERS') 
    plt.savefig('NTKS-0330/NTK-projection-max-%dsteps-%s.png'%(steps,doamin)) 
    o+=1
    for w in range(W):
        plt.figure(o,dpi=120)
        sns.heatmap(data=longtailed_NTK[:,w,:],
                    vmin=-1,
                    vmax=1,
                    cmap=plt.get_cmap('Blues')
            )
        plt.title('PROJECTIONS_IN_HEADS_EVERY_LAYERS') 
        plt.savefig('NTKS-0330/NTK-projection-%dsteps-%dW-%s.png'%(steps,w,doamin)) 
        o+=1
    plt.figure(o,dpi=120)
    sns.heatmap(data=torch.max(longtailed_sim[:,0:2,:],dim=1)[0],
                vmin=-1,
                vmax=1,
                cmap=plt.get_cmap('Blues')
        )
    plt.title('SIMILARITY_IN_HEADS_EVERY_LAYERS') 
    plt.savefig('NTKS-0330/NTK-similarity-max-%dsteps-%s.png'%(steps,doamin)) 
    o+=1
    for w in range(W):
        plt.figure(o,dpi=120)
        sns.heatmap(data=longtailed_sim[:,w,:],
                    vmin=-1,
                    vmax=1,
                    cmap=plt.get_cmap('Blues')
            )
        plt.title('SIMILARITY_IN_HEADS_EVERY_LAYERS') 
        plt.savefig('NTKS-0330/NTK-similarity-%dsteps-%dW-%s.png'%(steps,w,doamin)) 
        o+=1
    print(steps,"finished")
    return NTKs, NTK,lambdamax_layers,o

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


def validate(model, val_loader, accelerator):
    losses = []

    for i, batch in enumerate(val_loader):  
        # print("valid",batch["w_ids"])
        # batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
        # batch = {key: tensor.to(device) for key, tensor in batch.items()}

        # _,_,_,_,_,_,inputs = model0(**batch)
        # inputs = [i0.to(device) for i0 in inputs]
        with torch.no_grad():
            
            loss, _= model(batch['input_ids'],batch['attention_mask'], batch['labels'])
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

def train(model, num_epochs, dataset, device):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    accelerator = Accelerator()
    freze_emb = 0 
    AngeL = 0
    sparsity_rate = 0
    SPAR = 1
    orth_rate = 0
    ORTH = 1
    same_in_param = 0

    with_bert_param = 1

    # model01 = base_models.BertForMLM(config=config0)
    # model0 = base_models.BertForMLM_toshow(config=config0)
    # load_checkpoint_and_dispatch(model01, '/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0401/0401-BERT-3072ffns-wikipedia-1.5e-4-checkpoints70000',device_map="auto") 
    # # model01 = torch.load('0226-bert-batch64_seq128-MIXED_LEGAL_PUBMED-pretrain-lr2e-4-for-routing2.pth')

    # model0.load_state_dict(model01.state_dict())


    # cluster_centers = load_layer_data('0226-layer_centers-2t-MIXED_LEGAL_PUBMED2.pth')



    train_loader, val_loader= dataset.train_loader, dataset.val_loader
    # val_loader1 = dataset.val_loader_domain1
    # val_loader2 = dataset.val_loader_domain2
    # val_loader_domain1_ntk = dataset.val_loader_domain1_ntk
    # val_loader_domain2_ntk = dataset.val_loader_domain2_ntk
    # ahead_train_loader = ahead_dataset.train_loader
    # ahead_val_loader = ahead_dataset.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    
    # writer = SummaryWriter('tensorboard_0119/0119_MoMoE_batch64_seq128_4t16e_WIKI103_WP10_LAR3')
    writer = SummaryWriter('tensorboard_0514/MOMOE-T-max-multipleatt-768ffns-mixed-small-1.5e-4')
    # writer = SummaryWriter('tensorboard_0311/0311_MoMoE_sharedW_64*128_3t4e_LARGE_SPECIFIC_GENERAL-lr3e-4')

    
    # writer = SummaryWriter('0118_MoMo_CAU_v1_batch32_seq256-show')


    
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    # model,optimizer, lr_scheduler, train_loader, val_loader,val_loader1,val_loader2= accelerator.prepare(model,optimizer, lr_scheduler, train_loader, val_loader,val_loader1,val_loader2)
    # model.to(device)
    # center_model = base_models.BertForMLM(config=config)
    # load_checkpoint_and_dispatch(center_model, '/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0401/0401-BERT-3072ffns-wikipedia-1.5e-4-checkpoints7000',device_map={"": device}) 
     

    # model0.to(device2)
    # router.to(device)
    # if freze_emb:
    #     for para in model.embeddings.parameters():
    #         para.requires_grad = False
    
    ####for unique and common dimension#####
    # step0 = 0
    # VECTORS_PRO = []
    # model0.eval()
    
    # HI_OS = []
    # l = 4
    # for i, batch in enumerate(train_loader):
    #     step0 += 1
    #     print(i)
    #     if step0 == 35:
    #         break
    #     batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
        
    #     _,_,_,_,_,_,inputs = model0(**batch0)
        
        
    #     HI_OS.append(inputs[l].detach())
    # DATA_FOR_ROUTE = []
    
    # HI_OS=torch.cat(HI_OS)
    # data_class = get_clusters(HI_OS,config.num_transformer)
    # print(data_class[1].shape)
    # eig_vecs = get_gds(data_class)
    # # print(eig_vecs.shape)
    # torch.save(eig_vecs, '0229-layer%d_pro_vec-2t-MIXED_LEGAL_PUBMED.pth'%l)
    PRO_VECS = []
    # for l in range(config.num_hidden_layers):
    #     eig_vecs0 = torch.load('0229-layer%d_pro_vec-2t-MIXED_LEGAL_PUBMED.pth'%l, map_location='cuda')
    #     PRO_VECS.append(eig_vecs0)
    # print("2222",eig_vecs0.shape)
    # print(len(train_loader))
    # W_ids = torch.zeros(len(train_loader),64)

    # for i, batch in enumerate(train_loader):
    #     _,_,_,_,_,_,_,wids = model0(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers[-1])
    #     W_ids[i][:batch['input_ids'].shape[0]] = wids.detach()
    #     print(i)
    # print('train finish')
    # torch.save(W_ids, '0403-W_IDS-TRAIN.pth')
    # W_ids2 = torch.zeros(len(val_loader),64)

    # for i, batch in enumerate(val_loader):
    #     _,_,_,_,_,_,_,wids = model0(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers[-1])
    #     W_ids2[i][:batch['input_ids'].shape[0]] = wids.detach()
    #     # print(wids)
    # model0.enable_cpu_offload()
    
    
    # y = 0
    steps = 0
    accelerator.save_state('MODELS_0514/MOMOE-T-max-multipleatt-768ffns-mixed-small-1.5e-4 -checkpoints%d'%steps)

    losses = []
    # WI_Losses = []
    # FN_Losses = []
    # for i, batch in enumerate(val_loader_domain1_ntk):
        
    # #     batchl = {k: v[:1, :] for k, v in batch.items()}
    #     break
    print(len(train_loader))
    for epoch in range(num_epochs):
        
        model.train()
        
        
        
        # routes_history = [0 for i in range(4**12)]
        for i, batch in enumerate(train_loader):
            # print(i)
            # print(batch['w_ids'])
            # if i>=15000:break
            # _, _, _,y = get_ntk_by_layer(model,val_loader_domain1_ntk,cluster_centers,PRO_VECS,model0,device2,steps,"legal",batchl,y)
            # 
            steps+=1
            # batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
            # batch = {key: tensor.to(device) for key, tensor in batch.items()}

            
            # _,_,_,_,_,_,inputs = model0(**batch)

            # inputs = [i0.to(device) for i0 in inputs]
            # inputs = [inputs[-1].to(device) for i in range(12)]
            # _, _, lambdamax_layers_t1_d1 = get_ntk_by_layer(model,val_loader_domain1_ntk,cluster_centers,PRO_VECS,model0,device2,steps)
            
            

            # cluster_centers = [cluster_centers[-1].to(device) for i in range(12)]
            
            # _, _, lambdamax_layers_t1_d2 = get_ntk_by_layer(model,cluster_centers,inputs,PRO_VECS,val_loader_domain2_ntk)
            # print(lambdamax_layers_t1_d1[0].shape)
            # head_lmax_norm, expert_lmax_norm = get_ntk(model, val_loader1, accelerator,cluster_centers,PRO_VECS,model0,device2)
            
            # print(head_lmax_norm[0])

            loss, _ = model(batch['input_ids'],batch['attention_mask'], batch['labels'])
            # print(ids)
            # idd = 0
            # for j in ids[0]:

            #     if j>7:
            #         idd = 1
            #         break
            #     # else:
            #         # print(i,"safe")
            # if idd:
            #     print(i,"false")
            # else:
            #     print(i,"safe")
                
            
            # all_loss = loss
            # print(layers_o[7].shape)
            

            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()    
    
            if steps%100 == 0:
                # print(f"steps:{steps}")
                # print(ids)
        
                loss_train = torch.mean(torch.cat(losses)[:len(losses)])

                
                loss_valid = validate(model, val_loader, accelerator)
                # loss_valid1 = validate(model, val_loader1, accelerator)

                # loss_valid2 = validate(model, val_loader2, accelerator)

                
                
                if sparsity_rate and orth_rate:
                    loss_L1 = torch.mean(torch.cat(FN_Losses)[:len(FN_Losses)])
                    loss_ORTH = torch.mean(torch.cat(WI_Losses)[:len(WI_Losses)])
                    accelerator.print(f'Epoch:{epoch} ({i} Updates), steps:{steps}, Train Loss: {loss_train}, Valid Loss: {loss_valid}, L1 Loss: {loss_L1}, ORTH Loss:{loss_ORTH}')
                elif sparsity_rate and not orth_rate:
                    loss_L1 = torch.mean(torch.cat(FN_Losses)[:len(FN_Losses)])
                    # loss_ORTH = torch.mean(torch.cat(WI_Losses)[:len(WI_Losses)])
                    accelerator.print(f'Epoch:{epoch} ({i} Updates), steps:{steps}, Train Loss: {loss_train}, Valid Loss: {loss_valid}, L1 Loss: {loss_L1}')
                elif not sparsity_rate and orth_rate:
                    # loss_L1 = torch.mean(torch.cat(FN_Losses)[:len(FN_Losses)])
                    loss_ORTH = torch.mean(torch.cat(WI_Losses)[:len(WI_Losses)])
                    accelerator.print(f'Epoch:{epoch} ({i} Updates), steps:{steps}, Train Loss: {loss_train}, Valid Loss: {loss_valid}, ORTH Loss:{loss_ORTH}')
                else:
                    # loss_L1 = torch.mean(torch.cat(FN_Losses)[:len(FN_Losses)])
                    # loss_ORTH = torch.mean(torch.cat(WI_Losses)[:len(WI_Losses)])
                    accelerator.print(f'Epoch:{epoch} ({i} Updates), steps:{steps}, Train Loss: {loss_train}, Valid Loss: {loss_valid}')
                if accelerator.is_local_main_process:
                    writer.add_scalar('perplexity_train_epoch', loss_train, steps)
                    writer.add_scalar('perplexity_valid', loss_valid, steps)
                    # writer.add_scalar('leagl_perplexity_valid', loss_valid1, steps)
                    # writer.add_scalar('pubmed_perplexity_valid', loss_valid2, steps)
                    # writer.add_scalar('general_perplexity_valid', loss_valid3, steps)
                    # writer.add_scalar('perplexity_ahead', ahead_train, steps)
                    writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], steps)
                losses = []
                WI_Losses = []
                FN_Losses = []

            if steps%7000 == 0:
                accelerator.save_state('MODELS_0514/MOMOE-T-max-multipleatt-768ffns-mixed-small-1.5e-4-checkpoints%d'%steps)

            # if steps%2000 == 0 and steps>1000:
            #     _, _, _,y = get_ntk_by_layer(model,val_loader_domain1_ntk,cluster_centers,PRO_VECS,model0,device2,steps,"legal",batchl,y)
            #     _, _, _,y = get_ntk_by_layer(model,val_loader_domain2_ntk,cluster_centers,PRO_VECS,model0,device2,steps,"bio",batchl,y)
                
    
    # torch.save(model,'MODELS_0315/0330-MoMoE-full-dimension-ntk-longtail-both.pth')
    # torch.save(model,'0311_MoMoE_sharedW_64*128_3t4e_LARGE_SPECIFIC_GENERAL-lr3e-4.pth')
    accelerator.save_state('MODELS_0514/MOMOE-T-max-multipleatt-768ffns-mixed-small-1.5e-4')
    
    


    

if __name__ == "__main__":
    set_seed(42)
    
    config = BertConfig.from_json_file('config/MoMoE.json')
    config0 = BertConfig.from_json_file('config/bert.json')

    # config = BertConfig.from_json_file('config/AngeL_rose2.json')

    # dataset = RestaurantForLM_small(config=config)
    # dataset = MoMoE_MIXED_WIKI103_0124(config = config)
    # ahead_dataset = MoMoE_MIXED_WIKI103_0124(config = config)
    dataset = MoMoE_longtailed(config = config)
    # ahead_dataset = Wikipedia_Wids(config = config)
    torch.cuda.set_device(2)
    device = torch.device("cuda:2")
    # device2 = torch.device("cuda:2")

    # model = base_models.MoMoE_0412(config=config)
    model = base_models.MoMoE_T_0514_2(config=config)
    
    # model = base_models.MoMoE_0413_onlyffns(config=config)
    # model = base_models.MoMoE_0313(config=config)
    # model = base_models.MoMoE_0404_adl(config=config)
    # model = base_models.MoMoE_super_finegrained(config=config)


    
    # model = base_models.MoMoE_salary1062(config=config)
    # model = base_models.MoMoE_uniqueatt_commonffn(config=config)

    # model = base_models.MoMoE_CAU(config=config)
    # model = base_models.MoMoE_prenorm(config=config)




    # router = base_models.BertWithSavers(config=config)
    # router.to(device)
    # model.to(device)
    # model = nn.DataParallel(model)

    
    train(model=model, num_epochs=1, dataset=dataset, device=device)