import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import RestaurantForLM_small, MixedData,MixedData_stage1,Mixdata_1103,Mixdata_1115,Wikitxt103ForLM_1103,Wikitxt103ForLM_0102_rose,RestaurantforLM_0109,ACLForLM_0109
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import umap


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


def validate(model, val_loader, accelerator, device, centers, router):
    losses = []
    for i, batch in enumerate(val_loader):  
        batch = {key: tensor.to(device) for key, tensor in batch.items()} 
        hidden_states_for_router = []
        _, _, layer_outputs,_ = router(**batch)
                #
               
        hidden_states_for_router.append(router.bert.embeddings(batch['input_ids']))
        hidden_states_for_router = hidden_states_for_router  + layer_outputs[0:-1]
        with torch.no_grad():
            
            loss, _, _,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], centers, hidden_states_for_router)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
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

def show(model, num_epochs, dataset, device, ahead_dataset, router ):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    
    freze_emb = 0 
    AngeL = 0
    sparsity_rate = 10
    SPAR = 1

    model0 = torch.load('0108-bert-MixedData_1211-2000.pth')
    model_baseline = base_models.BertForMLM_toshow(config=config0)
    model_baseline0 =torch.load('0108-bert-MixedData_1211-20000.pth')
    model_baseline.load_state_dict(model_baseline0.state_dict())
    # cluster_centers = load_layer_data('layer_centers.pth')
    cluster_centers = load_layer_data('0102-layer_centers-2expert.pth')
    inbition_centers = load_layer_data('0108-angel-justinhibition-combineres-centers.pth')
    print(len(inbition_centers))
    print(len(inbition_centers[0]))
    print(inbition_centers[0][0].shape)
    # print(len(cluster_centers), cluster_centers[9].shape)
    # model_rose = torch.load('0102_AngeL_rose_model2-1_satge1.pth')
    # model_rose = torch.load('0102_AngeL_rose_model2_satge1.pth')
    model_rose = torch.load('0108-angel-justinhibition-combineres-MixedData_1211.pth')
    # model_rose = torch.load('0107-angel-justinhibition-combineres.pth')
    model.load_state_dict(model_rose.state_dict())

    router.bert.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    for i in range(config.num_hidden_layers):
        router.bert.layers.layers[i].load_state_dict(model0.bert.encoders.layers[i].state_dict())

    router.head.load_state_dict(model0.head.state_dict())

    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    ahead_val_loader = ahead_dataset.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()

    
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    # model, optimizer, lr_scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    model,router, optimizer, lr_scheduler, train_loader, val_loader, ahead_val_loader = accelerator.prepare(model,router, optimizer, lr_scheduler, train_loader, val_loader,ahead_val_loader)
    model.to(device)
    device2 = torch.device("cuda:6")
    model_baseline.to(device2)

    if freze_emb:
        for para in model.embeddings.parameters():
            para.requires_grad = False
    
    
        
    model.eval()
    ATT_OUTS = [[]for i in range(12)]
    FFN_OUTS = [[]for i in range(12)]
    OUTPUTS = [[]for i in range(12)]
    ATT_SELF = [[]for i in range(12)]
    CLUSTER = [[]for i in range(12)]

    for i, batch in enumerate(train_loader):
        if i%150 == 0:
            print(i)
            hidden_states_for_router = []
            batch = {key: tensor.to(device2) for key, tensor in batch.items()}
            
            _,_,outputs,att_outs,ffn_outputs,att_self = model_baseline(**batch)
            for j in range(12):
                # ATT_OUTS[j].append(att_outs[j])
                # FFN_OUTS[j].append(ffn_outputs[j])
                # OUTPUTS[j].append(outputs[j])
                ATT_SELF[j].append(att_self[j])
                CLUSTER[j].append(att_self[j])
    


    for j in range(12):
        # ATT_OUTS[j]=torch.cat(ATT_OUTS[j]).view(-1,768).mean(0)
        # FFN_OUTS[j]=torch.cat(FFN_OUTS[j]).view(-1,768).mean(0)
        # OUTPUTS[j]=torch.cat(OUTPUTS[j]).view(-1,768).mean(0)
        CLUSTER[j]=torch.cat(CLUSTER[j])
        ATT_SELF[j]=torch.cat(ATT_SELF[j])

    for j in range(12):
        # ATT_OUTS[j],_=torch.sort(ATT_OUTS[j]/ATT_OUTS[j].norm(), descending=True)
        # FFN_OUTS[j],_=torch.sort(FFN_OUTS[j]/FFN_OUTS[j].norm(), descending=True)
        # OUTPUTS[j],_=torch.sort(OUTPUTS[j]/OUTPUTS[j].norm(), descending=True)
        # print(ATT_SELF[j].view(-1,768).mean(0).min().data)
        ATT_SELF[j],_=torch.sort((ATT_SELF[j].view(-1,768).mean(0)-ATT_SELF[j].view(-1,768).mean(0).min().data)/(ATT_SELF[j].view(-1,768).mean(0).max().data-ATT_SELF[j].view(-1,768).mean(0).min().data), descending=True)
    for j in range(12):
        s = "ATT_SELF"
        plt.figure(j)
        plt.plot(ATT_SELF[j].cpu().detach().numpy(),label="BERT")
        # plt.title("Visualization of a Normalized and Sorted 768-Dimensional Tensor--%d layer %s"%(j,s))
        # plt.xlabel("Index")
        # plt.ylabel("Value")
        # plt.grid(True)
        # plt.savefig("0105-BERT/0105-%s%dlayer.png"%(s,j))
    for j in range(12):
        plt.figure(j+400)
        # print(CLUSTER[j][0].shape)
        A = CLUSTER[j].mean(1).cpu().detach().numpy()
        # print(A.shape)
        # A1 = CLUSTER[j][0].mean(1).cpu().detach().numpy()
        # A2 = CLUSTER[j][1].mean(1).cpu().detach().numpy()
        U = umap.UMAP(random_state=42).fit_transform(A)
        # U1 = U.transform(A1)
        # U2 = U.transform(A2)
        plt.scatter(U[:, 0], U[:, 1], s=5)
        # plt.scatter(U2[:, 0], U2[:, 1], label="1", s=5)
        plt.legend()
        plt.grid(True)
        plt.savefig("0109-BERT/CLUSTER-%s%dlayer.png"%(s,j))

    


    # ATT_OUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # FFN_OUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # OUTPUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # ATT_SELF = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    
    # CLUSTER = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    
    
    # ORTH = []
    # for i, batch in enumerate(train_loader):
    #     if i==150:
    #         print(i)
    #         hidden_states_for_router = []
    #         batch = {key: tensor.to(device) for key, tensor in batch.items()}
    #         # print(epoch, i)
    #         # print(next(model.parameters()).device)
    #         # for key, tensor in batch.items():
    #         #     print(f"{key} is on {tensor.device}")
    #         _, _, layer_outputs,_ = router(**batch)
            
    #         hidden_states_for_router.append(router.bert.embeddings(batch['input_ids']))
    #         hidden_states_for_router = hidden_states_for_router  + layer_outputs[0:-1]
    #         # print(len(hidden_states_for_router))

    #         # route = [1 for i in range(int(config.batch_size/2))]+[0 for i in range(int(config.batch_size/2))]
    #         # # print(route)
    #         # random.shuffle(route)
    #         # # print(route)
    #         # routes = [route for i in range(config.num_hidden_layers)]
    #         # for j in range(12):
    #         #     model.layers.layers[j].set_state(0)
    #         # print(i,model.layers.layers[0].state, model.layers.layers[0].k,len(model.layers.layers[0].out_for_router[0]))
    #         # loss,_, OUT_OF_ROUTER,ROUTES,router_labels,att_self = model(batch['input_ids'],batch['attention_mask'],batch['labels'],routes)
            
    #         _, _, _, outputs, _,att_outs,ffn_outputs,att_self = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers, hidden_states_for_router)
    #         # print(len(att_self))
    #         # print(len(att_self[0][0]))
    #         print(att_self[6][0].shape)
    #         for j in range(12):
    #             for e in range(config.num_experts):
    #                 # ATT_OUTS[j][e].append(att_outs[j][e])
    #                 # FFN_OUTS[j].append(ffn_outputs[j][0])
    #                 # OUTPUTS[j].append(outputs[j][0])
    #                 ATT_SELF[j][e].append(att_self[j][e])
    #                 CLUSTER[j][e].append(att_self[j][e])
            
    
    # for j in range(12):
    #     orth = 0
    #     c = 0
    #     for e1 in range(config.num_experts-1):
    #         A = torch.cat(ATT_SELF[j][e1]).view(-1,768).mean(0)
    #         for e2 in range(e1+1,config.num_experts):
    #             B = torch.cat(ATT_SELF[j][e2]).view(-1,768).mean(0)
    #             # print(B.shape)
    #             # print(orth)

    #             orth = orth+torch.dot(A / A.norm(),B / B.norm()).item()
    #             # c=+1
    #             # dot = torch.dot(A / A.norm(),B / B.norm()).item()
    #             # print(c)
    #     # print(orth)
    #     ORTH.append(orth)
        
    #     for e in range(config.num_experts):
    #         # ATT_OUTS[j][e]=torch.cat(ATT_OUTS[j][e]).view(-1,768).mean(0)
    #         # FFN_OUTS[j]=torch.cat(FFN_OUTS[j]).view(-1,768).mean(0)
    #         # OUTPUTS[j]=torch.cat(OUTPUTS[j]).view(-1,768).mean(0)
    #         ATT_SELF[j][e]=torch.cat(ATT_SELF[j][e])
    #         CLUSTER[j][e]=torch.cat(CLUSTER[j][e])


    # for j in range(12):
    #     for e in range(config.num_experts):
    #         # ATT_OUTS[j][e],_=torch.sort(ATT_OUTS[j][e]/ATT_OUTS[j][e].norm(), descending=True)
    #         # FFN_OUTS[j],_=torch.sort(FFN_OUTS[j]/FFN_OUTS[j].norm(), descending=True)
    #         # OUTPUTS[j],_=torch.sort(OUTPUTS[j]/OUTPUTS[j].norm(), descending=True)
    #         ATT_SELF[j][e],_=torch.sort((ATT_SELF[j][e].view(-1,768).mean(0)-ATT_SELF[j][e].view(-1,768).mean(0).min().data)/(ATT_SELF[j][e].view(-1,768).mean(0).max().data-ATT_SELF[j][e].view(-1,768).mean(0).min().data), descending=True)
    # for j in range(12):
    #     s = "ATT_SELF"
    #     plt.figure(j)
    #     plt.plot(ATT_SELF[j][1].cpu().detach().numpy(),label = "AngeL_rose_lowrank64_AVEL1loss")
    #     plt.title("Visualization of a Normalized and Sorted 768-Dimensional Tensor--%d layer %s"%(j,s))
    #     plt.xlabel("Index")
    #     plt.ylabel("Value")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig("0108-rose/%s%dlayer.png"%(s,j))



    # s = "ATT_SELF"
    # plt.figure(200)
    # plt.plot(ORTH,label = "AngeL_rose_lowrank64_AVEL1loss")
    # plt.title("Visualization of a Normalized and Sorted 768-Dimensional Tensor--%d layer %s"%(j,s))
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("0108-rose/ORTH-%s%dlayer.png"%(s,j))
    
    # for j in range(12):
    #     plt.figure(j+100)
    #     # print(CLUSTER[j][0].shape)
    #     A = torch.cat((CLUSTER[j][0].mean(1),CLUSTER[j][1].mean(1)),dim=0).cpu().detach().numpy()
    #     print(A.shape)
    #     A1 = CLUSTER[j][0].mean(1).cpu().detach().numpy()
    #     A2 = CLUSTER[j][1].mean(1).cpu().detach().numpy()
    #     U = umap.UMAP(random_state=42).fit(A)
    #     UU = U.transform(A)
    #     U1 = U.transform(A1)
    #     U2 = U.transform(A2)
    #     plt.scatter(U1[:, 0], U1[:, 1], label="0", s=5)
    #     plt.scatter(U2[:, 0], U2[:, 1], label="1", s=5)
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig("0108-rose/CLUSTER-%s%dlayer.png"%(s,j))

    #     plt.figure(j+500)
    #     plt.scatter(UU[:, 0], UU[:, 1],  s=5)

    #     plt.savefig("0108-rose/CLUSTER_ALL-%s%dlayer.png"%(s,j))



    ATT_OUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    FFN_OUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    OUTPUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    ATT_SELF = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    
    CLUSTER = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    
    
    ORTH = []
    for i, batch in enumerate(train_loader):
        if i==150:
            print(i)
            hidden_states_for_router = []
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            # print(epoch, i)
            # print(next(model.parameters()).device)
            # for key, tensor in batch.items():
            #     print(f"{key} is on {tensor.device}")
            # _, _, layer_outputs,_ = router(**batch)
            
            # hidden_states_for_router.append(router.bert.embeddings(batch['input_ids']))
            # hidden_states_for_router = hidden_states_for_router  + layer_outputs[0:-1]
            # print(len(hidden_states_for_router))

            route = [1 for i in range(int(config.batch_size/2))]+[0 for i in range(int(config.batch_size/2))]
            # print(route)
            random.shuffle(route)
            # print(route)
            routes = [route for i in range(config.num_hidden_layers)]
            for j in range(12):
                model.layers.layers[j].set_state(1)
                for e in range(config.num_experts):
                    print(inbition_centers[j][e].shape)
                    model.layers.layers[j].centers.append(inbition_centers[j][e])

            print(i,model.layers.layers[0].state, model.layers.layers[0].k,len(model.layers.layers[0].out_for_router[0]))
            loss,_, OUT_OF_ROUTER,ROUTES,router_labels,att_self = model(batch['input_ids'],batch['attention_mask'],batch['labels'],routes)
            
            # _, _, _, outputs, _,att_outs,ffn_outputs,att_self = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers, hidden_states_for_router)

            print(att_self[6][0].shape)
            for j in range(12):
                for e in range(config.num_experts):
                    # ATT_OUTS[j][e].append(att_outs[j][e])
                    # FFN_OUTS[j].append(ffn_outputs[j][0])
                    # OUTPUTS[j].append(outputs[j][0])
                    ATT_SELF[j][e].append(att_self[j][e])
                    CLUSTER[j][e].append(att_self[j][e])
            
    
    for j in range(12):
        orth = 0
        c = 0
        for e1 in range(config.num_experts-1):
            A = torch.cat(ATT_SELF[j][e1]).view(-1,768).mean(0)
            for e2 in range(e1+1,config.num_experts):
                B = torch.cat(ATT_SELF[j][e2]).view(-1,768).mean(0)
                # print(B.shape)
                # print(orth)

                orth = orth+torch.dot(A / A.norm(),B / B.norm()).item()
                # c=+1
                # dot = torch.dot(A / A.norm(),B / B.norm()).item()
                # print(c)
        # print(orth)
        ORTH.append(orth)
        
        for e in range(config.num_experts):
            # ATT_OUTS[j][e]=torch.cat(ATT_OUTS[j][e]).view(-1,768).mean(0)
            # FFN_OUTS[j]=torch.cat(FFN_OUTS[j]).view(-1,768).mean(0)
            # OUTPUTS[j]=torch.cat(OUTPUTS[j]).view(-1,768).mean(0)
            ATT_SELF[j][e]=torch.cat(ATT_SELF[j][e])
            CLUSTER[j][e]=torch.cat(CLUSTER[j][e])


    for j in range(12):
        for e in range(config.num_experts):
            # ATT_OUTS[j][e],_=torch.sort(ATT_OUTS[j][e]/ATT_OUTS[j][e].norm(), descending=True)
            # FFN_OUTS[j],_=torch.sort(FFN_OUTS[j]/FFN_OUTS[j].norm(), descending=True)
            # OUTPUTS[j],_=torch.sort(OUTPUTS[j]/OUTPUTS[j].norm(), descending=True)
            ATT_SELF[j][e],_=torch.sort((ATT_SELF[j][e].view(-1,768).mean(0)-ATT_SELF[j][e].view(-1,768).mean(0).min().data)/(ATT_SELF[j][e].view(-1,768).mean(0).max().data-ATT_SELF[j][e].view(-1,768).mean(0).min().data), descending=True)
    for j in range(12):
        s = "ATT_SELF"
        plt.figure(j)
        plt.plot(ATT_SELF[j][1].cpu().detach().numpy(),label = "AngeL_rose_lowrank64_AVEL1loss")
        plt.title("Visualization of a Normalized and Sorted 768-Dimensional Tensor--%d layer %s"%(j,s))
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig("0109-INHIBITION-MIXDATA/%s%dlayer.png"%(s,j))



    s = "ATT_SELF"
    plt.figure(200)
    plt.plot(ORTH,label = "AngeL_rose_lowrank64_AVEL1loss")
    plt.title("Visualization of a Normalized and Sorted 768-Dimensional Tensor--%d layer %s"%(j,s))
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("0109-INHIBITION-MIXDATA/ORTH-%s%dlayer.png"%(s,j))
    
    for j in range(12):
        plt.figure(j+100)
        # print(CLUSTER[j][0].shape)
        A = torch.cat((CLUSTER[j][0].mean(1),CLUSTER[j][1].mean(1)),dim=0).cpu().detach().numpy()
        print(A.shape)
        A1 = CLUSTER[j][0].mean(1).cpu().detach().numpy()
        A2 = CLUSTER[j][1].mean(1).cpu().detach().numpy()
        U = umap.UMAP(random_state=42).fit(A)
        UU = U.transform(A)
        U1 = U.transform(A1)
        U2 = U.transform(A2)
        plt.scatter(U1[:, 0], U1[:, 1], label="0", s=5)
        plt.scatter(U2[:, 0], U2[:, 1], label="1", s=5)
        plt.legend()
        plt.grid(True)
        plt.savefig("0109-INHIBITION-MIXDATA/CLUSTER-%s%dlayer.png"%(s,j))

        plt.figure(j+500)
        plt.scatter(UU[:, 0], UU[:, 1],  s=5)

        plt.savefig("0109-INHIBITION-MIXDATA/CLUSTER_ALL-%s%dlayer.png"%(s,j))
    

    


    

if __name__ == "__main__":
    set_seed(45)
    config0 = BertConfig.from_json_file('config/MoMoE.json')
    config = BertConfig.from_json_file('config/MoMoE.json')
    # dataset = RestaurantForLM_small(config=config)
    dataset = Wikitxt103ForLM_0102_rose(config = config)
    ahead_dataset = MixedData(config = config)
    
    device = torch.device("cuda:1")
    # model = base_models.AngeLB_MoE(config=config)
    # model = base_models.rose_model(config=config0)

    model = base_models.BertForMLM(config=config)

    router = base_models.BertWithSavers(config=config0)
    router.to(device)
    model.to(device)
    # model = nn.DataParallel(model)

    
    show(model=model, num_epochs=50, dataset=dataset, device=device, ahead_dataset = ahead_dataset, router = router)