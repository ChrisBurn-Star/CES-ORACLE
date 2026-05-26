import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import MixedData_1211,Wikitxt103ForLM_0109,RestaurantforLM_0109,ACLForLM_0109,RestaurantForLM_small, MixedData,MixedData_stage1,Mixdata_1103,Mixdata_1115,Wikitxt103ForLM_1103,Wikitxt103ForLM_0102_rose
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import umap
from transformer.Transformer_MOE import BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class BertForMLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
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
        output,ATT_SELF = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores,ATT_SELF

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

def show(model, num_epochs, dataset1, device, dataset2, router ):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    
    freze_emb = 0 
    AngeL = 0
    sparsity_rate = 10
    SPAR = 1

    model0 = torch.load('0108-bert-MixedData_1211-5000.pth')
    model_baseline = base_models.BertForMLM_toshow(config=config0)
    model_baseline_moe = BertForMLM(config = BertConfig.from_json_file('config/moe.json'))
    model_baseline0 =torch.load('0108-bert-MixedData_1211-20000.pth')
    model_baseline.load_state_dict(model_baseline0.state_dict())
    model_baseline0_moe =torch.load('0111-moe-form-yubin-mixdata20000.pth')
    model_baseline_moe.load_state_dict(model_baseline0_moe.state_dict())
    # cluster_centers = load_layer_data('layer_centers.pth')
    cluster_centers = load_layer_data('0108-layer_centers-2expert-MixedData_1211-5000.pth')
    inbition_centers = load_layer_data('0108-angel-justinhibition-combineres-centers-withoutupdatacenters.pth')
    # print(len(inbition_centers))
    # print(len(inbition_centers[0]))
    # print(inbition_centers[0][0].shape)
    # print(len(cluster_centers), cluster_centers[9].shape)
    # model_rose = torch.load('0102_AngeL_rose_model2-1_satge1.pth')
    # model_rose = torch.load('0102_AngeL_rose_model2_satge1.pth')
    model_rose = torch.load('0115_AngeL_rose_model_rank_128-MixedData_1211-better-centers.pth')
    # model_rose = torch.load('0107-angel-justinhibition-combineres.pth')
    model.load_state_dict(model_rose.state_dict())

    router.bert.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    for i in range(config.num_hidden_layers):
        router.bert.layers.layers[i].load_state_dict(model0.bert.encoders.layers[i].state_dict())

    router.head.load_state_dict(model0.head.state_dict())

    train_loader1, val_loader = dataset1.train_loader, dataset1.val_loader
    train_loader2 = dataset2.train_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()


    
    num_updates = num_epochs * len(train_loader1)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    # model, optimizer, lr_scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    model,router, optimizer, lr_scheduler, train_loader1,train_loader2= accelerator.prepare(model,router, optimizer, lr_scheduler, train_loader1,train_loader2)
    model.to(device)
    device2 = torch.device("cuda:6")
    model_baseline.to(device2)
    model_baseline_moe.to(device2)

    # FNNP1 = sum([p.abs().mean() for p in model.layers.layers[1].experts[0].parameters()])
    # FNNP2 = sum([p.abs().mean() for p in model.layers.layers[1].experts[1].parameters()])


    # print(FNNP1,FNNP2)

    if freze_emb:
        for para in model.embeddings.parameters():
            para.requires_grad = False
    
    
        
    model.eval()
    ATT_OUTS = [[]for i in range(12)]
    FFN_OUTS = [[]for i in range(12)]
    OUTPUTS = [[]for i in range(12)]
    ATT_SELF = [[[]for j in range(2)]for i in range(12)]
    CLUSTER = [[[]for j in range(2)]for i in range(12)]

    for i, batch in enumerate(train_loader1):
        
        hidden_states_for_router = []
        batch = {key: tensor.to(device2) for key, tensor in batch.items()}
        
        # _,_,outputs,att_outs,ffn_outputs,att_self = model_baseline(**batch)
        _,_,att_self = model_baseline_moe(**batch)
        for j in range(12):
            # ATT_OUTS[j].append(att_outs[j])
            # FFN_OUTS[j].append(ffn_outputs[j])
            # OUTPUTS[j].append(outputs[j])
            ATT_SELF[j][0].append(att_self[j])
            CLUSTER[j][0].append(att_self[j])


    for i, batch in enumerate(train_loader2):
        
        hidden_states_for_router = []
        batch = {key: tensor.to(device2) for key, tensor in batch.items()}
        
        # _,_,outputs,att_outs,ffn_outputs,att_self = model_baseline(**batch)
        _,_,att_self = model_baseline_moe(**batch)
        for j in range(12):
            # ATT_OUTS[j].append(att_outs[j])
            # FFN_OUTS[j].append(ffn_outputs[j])
            # OUTPUTS[j].append(outputs[j])
            ATT_SELF[j][1].append(att_self[j])
            CLUSTER[j][1].append(att_self[j])
    


    for j in range(12):
        # ATT_OUTS[j]=torch.cat(ATT_OUTS[j]).view(-1,768).mean(0)
        # FFN_OUTS[j]=torch.cat(FFN_OUTS[j]).view(-1,768).mean(0)
        # OUTPUTS[j]=torch.cat(OUTPUTS[j]).view(-1,768).mean(0)
        CLUSTER[j][0]=torch.cat(CLUSTER[j][0])
        ATT_SELF[j][0]=torch.cat(ATT_SELF[j][0])
        CLUSTER[j][1]=torch.cat(CLUSTER[j][1])
        ATT_SELF[j][1]=torch.cat(ATT_SELF[j][1])

    for j in range(12):
        # ATT_OUTS[j],_=torch.sort(ATT_OUTS[j]/ATT_OUTS[j].norm(), descending=True)
        # FFN_OUTS[j],_=torch.sort(FFN_OUTS[j]/FFN_OUTS[j].norm(), descending=True)
        # OUTPUTS[j],_=torch.sort(OUTPUTS[j]/OUTPUTS[j].norm(), descending=True)
        # print(ATT_SELF[j].view(-1,768).mean(0).min().data)
        ATT_SELF[j][0],_=torch.sort((ATT_SELF[j][0].view(-1,768).mean(0)-ATT_SELF[j][0].view(-1,768).mean(0).min().data)/(ATT_SELF[j][0].view(-1,768).mean(0).max().data-ATT_SELF[j][0].view(-1,768).mean(0).min().data), descending=True)
        ATT_SELF[j][1],_=torch.sort((ATT_SELF[j][1].view(-1,768).mean(0)-ATT_SELF[j][1].view(-1,768).mean(0).min().data)/(ATT_SELF[j][1].view(-1,768).mean(0).max().data-ATT_SELF[j][1].view(-1,768).mean(0).min().data), descending=True)

    
    # for j in range(12):
    #     s = "ATT_SELF"
    #     plt.figure(j)
    #     plt.plot(torch.cat((ATT_SELF[j][0],ATT_SELF[j][1]),dim=0).cpu().detach().numpy(),label="BERT")
    #     # plt.title("Visualization of a Normalized and Sorted 768-Dimensional Tensor--%d layer %s"%(j,s))
    #     # plt.xlabel("Index")
    #     # plt.ylabel("Value")
    #     # plt.grid(True)
    #     # plt.savefig("0105-BERT/0105-%s%dlayer.png"%(s,j))
    for j in range(12):
        s = "ATT_SELF"
        plt.figure(j+400)
        print(CLUSTER[j][0].shape)
        A = torch.cat((CLUSTER[j][0].mean(1),CLUSTER[j][1].mean(1)),dim=0).cpu().detach().numpy()
        # print(A.shape)
        A1 = CLUSTER[j][0].mean(1).cpu().detach().numpy()
        A2 = CLUSTER[j][1].mean(1).cpu().detach().numpy()
        U = umap.UMAP(random_state=42).fit(A)
        UU = U.transform(A)
        U1 = U.transform(A1)
        U2 = U.transform(A2)
        plt.scatter(U1[:, 0], U1[:, 1], label="rea", s=5)
        plt.scatter(U2[:, 0], U2[:, 1], label="acl", s=5)
        plt.legend()
        plt.grid(True)
        plt.savefig("0116-MoE/CLUSTER-%s%dlayer.png"%(s,j))

    


    # ATT_OUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # FFN_OUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # OUTPUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # ATT_SELF = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    
    # CLUSTER = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    
    # t1 = [[0 for i in range(2)] for j in range(12)]
    # ORTH = []
    # for i, batch in enumerate(train_loader1):

    #     print(i)
    #     hidden_states_for_router = []
    #     batch = {key: tensor.to(device) for key, tensor in batch.items()}
    #     # print(epoch, i)
    #     # print(next(model.parameters()).device)
    #     # for key, tensor in batch.items():
    #     #     print(f"{key} is on {tensor.device}")
    #     _, _, layer_outputs,_ = router(**batch)
        
    #     hidden_states_for_router.append(router.bert.embeddings(batch['input_ids']))
    #     hidden_states_for_router = hidden_states_for_router  + layer_outputs[0:-1]
    #     # print(len(hidden_states_for_router))

    #     # route = [1 for i in range(int(config.batch_size/2))]+[0 for i in range(int(config.batch_size/2))]
    #     # # print(route)
    #     # random.shuffle(route)
    #     # # print(route)
    #     # routes = [route for i in range(config.num_hidden_layers)]
    #     # for j in range(12):
    #     #     model.layers.layers[j].set_state(0)
    #     # print(i,model.layers.layers[0].state, model.layers.layers[0].k,len(model.layers.layers[0].out_for_router[0]))
    #     # loss,_, OUT_OF_ROUTER,ROUTES,router_labels,att_self = model(batch['input_ids'],batch['attention_mask'],batch['labels'],routes)
        
    #     _, _, _, outputs, _,att_outs,ffn_outputs,att_self = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers, hidden_states_for_router)
    #     # print(len(att_self))
    #     # print(len(att_self[0][0]))
    #     # print(att_self[6][0].shape)
    #     for j in range(12):
    #         for e in range(config.num_experts):
    #             # ATT_OUTS[j][e].append(att_outs[j][e])
    #             # FFN_OUTS[j].append(ffn_outputs[j][0])
    #             # OUTPUTS[j].append(outputs[j][0])
    #             # ATT_SELF[j][e].append(att_self[j][e])
    #             # CLUSTER[j][e].append(att_self[j][e])
    #             if len(att_self[j][e]):
    #                 print("layer",j,"expert",e)
    #                 ATT_SELF[j][e].append(att_self[j][e])
    #                 CLUSTER[j][e].append(att_self[j][e])
    #                 t1[j][e]+=att_self[j][e].shape[0]


    # for i, batch in enumerate(train_loader2):
        
    #     print(i)
    #     hidden_states_for_router = []
    #     batch = {key: tensor.to(device) for key, tensor in batch.items()}
    #     # print(epoch, i)
    #     # print(next(model.parameters()).device)
    #     # for key, tensor in batch.items():
    #     #     print(f"{key} is on {tensor.device}")
    #     _, _, layer_outputs,_ = router(**batch)
        
    #     hidden_states_for_router.append(router.bert.embeddings(batch['input_ids']))
    #     hidden_states_for_router = hidden_states_for_router  + layer_outputs[0:-1]
    #     # print(len(hidden_states_for_router))

    #     # route = [1 for i in range(int(config.batch_size/2))]+[0 for i in range(int(config.batch_size/2))]
    #     # # print(route)
    #     # random.shuffle(route)
    #     # # print(route)
    #     # routes = [route for i in range(config.num_hidden_layers)]
    #     # for j in range(12):
    #     #     model.layers.layers[j].set_state(0)
    #     # print(i,model.layers.layers[0].state, model.layers.layers[0].k,len(model.layers.layers[0].out_for_router[0]))
    #     # loss,_, OUT_OF_ROUTER,ROUTES,router_labels,att_self = model(batch['input_ids'],batch['attention_mask'],batch['labels'],routes)
        
    #     _, _, _, outputs, _,att_outs,ffn_outputs,att_self = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers, hidden_states_for_router)
    #     # print(len(att_self))
    #     # print(len(att_self[0][0]))
    #     # print(att_self[6][0].shape)
    #     for j in range(12):
    #         for e in range(config.num_experts):
    #             # ATT_OUTS[j][e].append(att_outs[j][e])
    #             # FFN_OUTS[j].append(ffn_outputs[j][0])
    #             # OUTPUTS[j].append(outputs[j][0])
    #             # ATT_SELF[j][e].append(att_self[j][e])
    #             # CLUSTER[j][e].append(att_self[j][e])
    #             if len(att_self[j][e]):
    #                 print("layer",j,"expert",e)
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
    #     plt.savefig("0111_AngeL_rose_model_rank_128-MixedData_1211-better-centers-pngs/%s%dlayer.png"%(s,j))



    # s = "ATT_SELF"
    # plt.figure(200)
    # plt.plot(ORTH,label = "AngeL_rose_lowrank64_AVEL1loss")
    # plt.title("Visualization of a Normalized and Sorted 768-Dimensional Tensor--%d layer %s"%(j,s))
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("0111_AngeL_rose_model_rank_128-MixedData_1211-better-centers-pngs/ORTH-%s%dlayer.png"%(s,j))
    
    # for j in range(12):
    #     print()
    #     plt.figure(j+100)
    #     # print(CLUSTER[j][0].shape)
    #     A = torch.cat((CLUSTER[j][0].mean(1),CLUSTER[j][1].mean(1)),dim=0).cpu().detach().numpy()
    #     print(A.shape)
    #     A1 = CLUSTER[j][0].mean(1).cpu().detach().numpy()
    #     print(A1.shape)
    #     A2 = CLUSTER[j][1].mean(1).cpu().detach().numpy()
    #     print(A2.shape)
    #     U = umap.UMAP(random_state=42).fit(A)
    #     print("layer",j,"res",t1[j][0]+t1[j][1],"res in expert1",t1[j][0])
    #     UU = U.transform(A)
    #     U1 = U.transform(A1)
    #     U2 = U.transform(A2)
    #     plt.scatter(U1[:t1[j][0], 0], U1[:t1[j][0], 1],label="res_in_expert1", s=5)
    #     plt.scatter(U2[:t1[j][1], 0], U2[:t1[j][1], 1],label="res_in_expert2", s=5)

    #     plt.scatter(U1[t1[j][0]:, 0], U1[t1[j][0]:, 1],label="acl_in_expert1", s=5)
    #     plt.scatter(U2[t1[j][1]:, 0], U2[t1[j][1]:, 1],label="acl_in_expert2", s=5)
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig("0111_AngeL_rose_model_rank_128-MixedData_1211-better-centers-pngs/CLUSTER-%s%dlayer.png"%(s,j))

    #     plt.figure(j+500)
    #     plt.scatter(UU[:, 0], UU[:, 1],  s=5)

    #     plt.savefig("0111_AngeL_rose_model_rank_128-MixedData_1211-better-centers-pngs/CLUSTER_ALL-%s%dlayer.png"%(s,j))



    # ATT_OUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # FFN_OUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # OUTPUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # ATT_SELF = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    
    # CLUSTER = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    
    
    # ORTH = []
    # t1 = [[0 for i in range(2)] for j in range(12)]
    # for i, batch in enumerate(train_loader1):
    #     print("i",i)
    #     hidden_states_for_router = []
    #     batch = {key: tensor.to(device) for key, tensor in batch.items()}

    #     route = [1 for i in range(int(config.batch_size/2))]+[0 for i in range(int(config.batch_size/2))]
    #     # print(route)
    #     random.shuffle(route)
    #     # print(route)
    #     routes = [route for i in range(config.num_hidden_layers)]
    #     if i == 0:
    #         for j in range(12):
    #             model.layers.layers[j].set_state(1)
    #             # print(inbition_centers[j][0]-inbition_centers[j][1],inbition_centers[j][0])
    #             for e in range(config.num_experts):
    #                 # print(inbition_centers[j][e].shape)
    #                 model.layers.layers[j].centers.append(inbition_centers[j][e])
            

    #     print(i,model.layers.layers[0].state, model.layers.layers[0].k,len(model.layers.layers[0].out_for_router[0]))
    #     loss,_, OUT_OF_ROUTER,ROUTES,router_labels,att_self = model(batch['input_ids'],batch['attention_mask'],batch['labels'],routes)
        
    #     # _, _, _, outputs, _,att_outs,ffn_outputs,att_self = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers, hidden_states_for_router)

    #     print(ROUTES)
    #     for j in range(12):
    #         for e in range(config.num_experts):
    #             # ATT_OUTS[j][e].append(att_outs[j][e])
    #             # FFN_OUTS[j].append(ffn_outputs[j][0])
    #             # OUTPUTS[j].append(outputs[j][0])
    #             if len(att_self[j][e]):
    #                 print("layer",j,"expert",e)
    #                 ATT_SELF[j][e].append(att_self[j][e])
    #                 CLUSTER[j][e].append(att_self[j][e])
    #                 t1[j][e]+=att_self[j][e].shape[0]

    
    # t2 = [0 for i in range(2)]
    # for i, batch in enumerate(train_loader2):
    #     print("i",i)
    #     hidden_states_for_router = []
    #     batch = {key: tensor.to(device) for key, tensor in batch.items()}

    #     route = [1 for i in range(int(config.batch_size/2))]+[0 for i in range(int(config.batch_size/2))]
    #     # print(route)
    #     random.shuffle(route)
    #     # print(route)
    #     routes = [route for i in range(config.num_hidden_layers)]
    #     # print(inbition_centers[0][0]==inbition_centers[0][1])
    #     for j in range(12):
    #         model.layers.layers[j].set_state(1)
    #         for e in range(config.num_experts):
    #             # print(inbition_centers[j][e].shape)
    #             model.layers.layers[j].centers.append(inbition_centers[j][e])

    #     print(i,model.layers.layers[0].state, model.layers.layers[0].k,len(model.layers.layers[0].out_for_router[0]))
    #     loss,_, OUT_OF_ROUTER,ROUTES,router_labels,att_self = model(batch['input_ids'],batch['attention_mask'],batch['labels'],routes)
        
    #     # _, _, _, outputs, _,att_outs,ffn_outputs,att_self = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers, hidden_states_for_router)
    #     print(ROUTES)
    #     print(att_self[6][0].shape)
    #     for j in range(12):
    #         for e in range(config.num_experts):
    #             # ATT_OUTS[j][e].append(att_outs[j][e])
    #             # FFN_OUTS[j].append(ffn_outputs[j][0])
    #             # OUTPUTS[j].append(outputs[j][0])
    #             if len(att_self[j][e]):
    #                 print("layer",j,"expert",e)
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
    #     plt.savefig("0109-INHIBITION-MIXDATA/%s%dlayer.png"%(s,j))



    # s = "ATT_SELF"
    # plt.figure(200)
    # plt.plot(ORTH,label = "AngeL_rose_lowrank64_AVEL1loss")
    # plt.title("Visualization of a Normalized and Sorted 768-Dimensional Tensor--%d layer %s"%(j,s))
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("0109-INHIBITION-MIXDATA/ORTH-%s%dlayer.png"%(s,j))
    
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
    #     plt.scatter(U1[:t1[j][0], 0], U1[:t1[j][0], 1],label="res_in_expert1", s=5)
    #     plt.scatter(U2[:t1[j][1], 0], U2[:t1[j][1], 1],label="res_in_expert2", s=5)

    #     plt.scatter(U1[t1[j][0]:, 0], U1[t1[j][0]:, 1],label="acl_in_expert1", s=5)
    #     plt.scatter(U2[t1[j][1]:, 0], U2[t1[j][1]:, 1],label="acl_in_expert2", s=5)
        
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig("0109-INHIBITION-MIXDATA/CLUSTER-%s%dlayer.png"%(s,j))

    #     plt.figure(j+500)
    #     plt.scatter(UU[:, 0], UU[:, 1],  s=5)

    #     plt.savefig("0109-INHIBITION-MIXDATA/CLUSTER_ALL-%s%dlayer.png"%(s,j))
    

    


    

if __name__ == "__main__":
    set_seed(45)
    config0 = BertConfig.from_json_file('config/AngeL_rose.json')
    config = BertConfig.from_json_file('config/AngeL_MoE.json')
    # dataset = RestaurantForLM_small(config=config)
    dataset1 = RestaurantforLM_0109(config = config)
    dataset2 = ACLForLM_0109(config = config)
    
    device = torch.device("cuda")
    model = base_models.AngeLB_MoE(config=config)
    model = base_models.AngeL_rose_model(config=config0)

    # model = base_models.BertForMLM(config=config)

    router = base_models.BertWithSavers(config=config0)
    router.to(device)
    model.to(device)
    # model = nn.DataParallel(model)

    
    show(model=model, num_epochs=50, dataset1=dataset1, device=device, dataset2 = dataset2, router = router)