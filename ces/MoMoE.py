import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import MoMoE_LARGE_SPECIFIC_GENERAL,MoMoE_MIXED_LEGAL_PUBMED,MoMoE_MIXED_0128,MoMoE_MIXED_WIKI103_0124,MoMoE_MIXED,MoMoE_WIKI103,MixedData_1211,MixedData_0110_1,MixedData_1211_1,RestaurantForLM_small, MixedData,MixedData_stage1,Mixdata_1103,Mixdata_1115,Wikitxt103ForLM_1103,Wikitxt103ForLM_0102_rose,MoMoE_FEWER_SPECIFIC_GENERAL
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
            
            loss, _, _, _, _,_,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], centers,inputs,PRO_VERS)
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
    model01 = torch.load('0226-bert-batch64_seq128-MIXED_LEGAL_PUBMED-pretrain-lr2e-4-for-routing2.pth')
    model0.load_state_dict(model01.state_dict())

    cluster_centers = load_layer_data('0229-layer_centers-2t-MIXED_LEGAL_PUBMED2-unique.pth')


    train_loader, val_loader = dataset.train_loader, dataset.val_loader

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
 
    writer = SummaryWriter('tensorboard_0311/0311_MoMoE_GDS_64*128_2t4e_LARGE_SPECIFIC_GENERAL-lr3e-4')

    
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)

    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    model.to(device)

    device2 = torch.device("cuda:6")
    model0.to(device2) 

    if freze_emb:
        for para in model.embeddings.parameters():
            para.requires_grad = False
    


    model0.eval()
    
    
    # for l in range(12):
    #     HI_OS = []
    #     step0 = 0
    #     for i, batch in enumerate(train_loader):
    #         step0 += 1
    #         print(i)
    #         if step0 == 120:
    #             break
    #         batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
            
    #         _,_,_,_,_,_,inputs = model0(**batch0)
            
            
    #         HI_OS.append(inputs[l].detach())
    #     DATA_FOR_ROUTE = []
        
    #     HI_OS=torch.cat(HI_OS)
    #     data_class = get_clusters(HI_OS,config.num_transformer)
    #     print(data_class[1].shape)
    #     eig_vecs = get_gds(data_class)
    #     # print(eig_vecs.shape)
    #     torch.save(eig_vecs, '0311-layer%d_pro_vec-3t-MIXED_LEGAL_PUBMED.pth'%l)
    # print("saved")
    PRO_VECS = []
    for l in range(config.num_hidden_layers):
        eig_vecs0 = torch.load('0229-layer%d_pro_vec-2t-MIXED_LEGAL_PUBMED.pth'%l, map_location='cuda')
        PRO_VECS.append(eig_vecs0)
    # print("2222",eig_vecs0.shape)


        
    steps = 0
    losses = []
    WI_Losses = []
    FN_Losses = []
    for epoch in range(num_epochs):
        
        model.train()
        
        
        
        # routes_history = [0 for i in range(4**12)]
        for i, batch in enumerate(train_loader):
            steps+=1
            batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            # print(epoch, i)
            # # print(next(model.parameters()).device)
            # # for key, tensor in batch.items():
            # #     print(f"{key} is on {tensor.device}")
            # # _, _, layer_outputs,_ = router(**batch)
            
            
            _,_,_,_,_,_,inputs = model0(**batch0)

            inputs = [i0.to(device) for i0 in inputs]
            
            loss, _, _, _, ids,_,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers,inputs,PRO_VECS)



            all_loss = loss

            if sparsity_rate and i%sparsity_rate == 0:
                
                f_c = 0
                for l in range(config.num_hidden_layers):
                    FN_loss = 0.0
                    for exp in range(config.num_experts):
                        # FNNP1 = sum([p.abs().mean() for p in model.layers[l].experts[exp].ffn.parameters()])
                        FNNP2 = sum([p.abs().mean() for p in model.layers[l].experts[exp].parameters()])
                        # FNNP = FNNP1+FNNP2
                        FNNP = FNNP2
                        FN_loss = FN_loss+ FNNP
                        all_loss = all_loss+1/(config.num_hidden_layers)*FN_loss
                        FN_Losses.append(accelerator.gather(FN_loss.repeat(config.batch_size)))
                        
                    
                # FN_loss = FN_loss/f_c
                # loss = loss + SPAR*FN_loss

            if orth_rate and i%orth_rate == 0:
                
                for l in range(config.num_hidden_layers):
                    WI_loss = 0.0
                    w_c = 0
                    for exp1 in range(config.num_experts-1):
                        for exp2 in range(exp1+1,config.num_experts):
                            
                            for head in range(config.num_attention_heads):
                                Wi0 = model.layers[l].experts[exp1].attention.self.heads[head].weight
                                # print(Wi0[1].shape)
                                Wi0 = Wi0.view(-1)/Wi0.view(-1).norm()

                                Wj0 = model.layers[l].experts[exp2].attention.self.heads[head].weight
                                Wj0 = Wj0.view(-1)/Wj0.view(-1).norm()
                                # print(Wi0.shape)


                                WI_loss = WI_loss+(Wi0*Wj0).abs().sum()
                                # print("Wi0*Wj0",(Wi0*Wj0).sum())
                                w_c = w_c+1
                                # print(WI_loss)
                            # Wi1 = model.layers[l].experts[exp1].attention.self.WO.weight.view(-1)
                            # Wi1 = Wi1.view(-1)/Wi1.view(-1).norm()
                            # Wj1 = model.layers[l].experts[exp2].attention.self.WO.weight.view(-1)
                            # Wj1 = Wj1.view(-1)/Wj1.view(-1).norm()
                            
                            # WI_loss = WI_loss+(Wi1*Wj1).abs().sum()
                            # w_c = w_c+1
                    WI_loss = WI_loss/w_c
                    all_loss = all_loss+1/(config.num_hidden_layers)*WI_loss
                    WI_Losses.append(accelerator.gather(WI_loss.repeat(config.batch_size)))
                    # print(WI_loss)


            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            optimizer.zero_grad()
            accelerator.backward(all_loss)
            optimizer.step()
            lr_scheduler.step()    

            if steps%100 == 0:
                print(f"steps:{steps}")
                print(ids)
        
                loss_train = torch.mean(torch.cat(losses)[:len(losses)])

                
                loss_valid = validate(model, val_loader, accelerator, device, cluster_centers,model0,device2,PRO_VECS)

                # loss_valid = validate(model, val_loader, accelerator, device, cluster_centers,model0,device2)
                # loss_valid = validate(model, val_loader, accelerator, device, cluster_centers)
                # loss_valid = validate(model, val_loader, accelerator, device)
                
                # ahead_train = validate(model, ahead_val_loader, accelerator, device, cluster_centers,router)
                # routes_history = torch.tensor(routes_history)
                # listss = torch.argsort(routes_history,descending=True)[:10]
                # loss_test = validate(model, test_loader, accelerator)
                # accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Test Loss: {loss_test}')
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
                    # writer.add_scalar('perplexity_ahead', ahead_train, steps)
                    writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], steps)
                losses = []
                WI_Losses = []
                FN_Losses = []
            # if steps%100000 == 0 and steps>0:
            #     # torch.save(model,'0119_MoMoE_batch64_seq128_4t16e_WIKI103_WP10_LAR3_%dsteps.pth'%steps)
            #     torch.save(model,'0219_MoMoE_uniqueatt_commonffn_batch16_seq256_2t8e_MIXED_LEGAL_PUBMED_autoinit_WP128000_%dsteps.pth'%steps)
                
                # torch.save(model,'0118_MoMoE_CAU_batch32_seq256_v1_%dsteps.pth'%steps)

    # accelerator.save_state('./output-formal-1027-new_model-stage1-freeze_embed')
    # torch.save(model,'0119_MoMoE_batch64_seq128_4t16e_WIKI103_WP5_inparams.pth')
    # torch.save(model,'0119_MoMoE_batch64_seq128_4t16e_WIKI103_WP10_LAR3.pth')
    torch.save(model,'0311_MoMoE_GDS_64*128_2t4e_LARGE_SPECIFIC_GENERAL-lr3e-4.pth')
    # CENTERS = {}
    # for l in range(config.num_hidden_layers):
    #     CENTERS[l] = model.layers[l].cluster_centers
    # torch.save(CENTERS,'0226_MoMoE_uniqueatt_commonffn_batch64_seq128_2t4e_MIXED_LEGAL_PUBMED_autoinit_WP128000-pretrain-lr2e-4-correct.pth')

    # torch.save(model,'0119_MoMoE_CAU_batch32_seq256_4t16e.pth')

    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/MoMoE.json')
    config0 = BertConfig.from_json_file('config/bert.json')

    # config = BertConfig.from_json_file('config/AngeL_rose2.json')

    # dataset = RestaurantForLM_small(config=config)
    # dataset = MoMoE_MIXED_WIKI103_0124(config = config)
    # ahead_dataset = MoMoE_MIXED_WIKI103_0124(config = config)
    dataset = MoMoE_LARGE_SPECIFIC_GENERAL(config = config)
    ahead_dataset = MoMoE_LARGE_SPECIFIC_GENERAL(config = config)
    torch.cuda.set_device(5)
    device = torch.device("cuda")
    model = base_models.MoMoE_0229(config=config)
    # model = base_models.MoMoE_salary1062(config=config)
    # model = base_models.MoMoE_uniqueatt_commonffn(config=config)

    # model = base_models.MoMoE_CAU(config=config)
    # model = base_models.MoMoE_prenorm(config=config)




    # router = base_models.BertWithSavers(config=config)
    # router.to(device)
    # model.to(device)
    # model = nn.DataParallel(model)

    
    train(model=model, num_epochs=1, dataset=dataset, device=device, ahead_dataset = ahead_dataset)