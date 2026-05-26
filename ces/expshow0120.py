import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import MixedData_1121,MoMoE_MIXED_WIKI103_0124_L,MoMoE_MIXED_WIKI103_0124_R,MoMoE_MIXED_WIKI103_0124_W,MoMoE_MIXED_TOSHOW,MoMoE_WIKI103_TOSHOW,MixedData_1211,Wikitxt103ForLM_0109,RestaurantforLM_0109,ACLForLM_0109,RestaurantForLM_small, MixedData,MixedData_stage1,Mixdata_1103,Mixdata_1115,Wikitxt103ForLM_1103,Wikitxt103ForLM_0102_rose
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
        output,ATT_SELF,O = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores,ATT_SELF,O

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


def load_layer_data(path,device):
    layer_data_dict = torch.load(path, map_location=device)
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

def show(model, num_epochs, dataset1, device, dataset2,dataset3, router ):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    
    freze_emb = 0 
    AngeL = 0
    sparsity_rate = 10
    SPAR = 1

    
    m1 = "MoE"
    
    # m1 = "BERT"

    # m2 = "MoT"

    # m2 = "MoT-MoE"

    # model_baseline = base_models.BertForMLM_toshow(config=config0)
    model_baseline_moe = BertForMLM(config = BertConfig.from_json_file('config/MoMoE.json'))  
    # model_baseline0 =torch.load('0119-bert-MIXED-ALL.pth')  #bert
    # model_baseline0 =torch.load('0119-bert-WIKI103-ALL.pth')  #bert

    # model_baseline.load_state_dict(model_baseline0.state_dict())
    model_baseline0_moe =torch.load('0119-moe-form-yubin-WIKI103-ALL.pth')
    # model_baseline0_moe =torch.load('0119-moe-form-yubin-MIXED-ALL.pth') #moe
    #  #moe
    model_baseline_moe.load_state_dict(model_baseline0_moe.state_dict())



  
    # cluster_centers = load_layer_data('0119-layer_centers-4t-WIKI103-WP10.pth',device)
    cluster_centers = load_layer_data('0119-layer_centers-4t-MIXED-WP3.pth',device)
    
    # model_rose = torch.load('0119_AngeL_rose_model_4expert_seq128_batchsize64_WP5_new.pth')
    # model_rose = torch.load('0119_MoMoE_batch64_seq128_4t16e.pth')
    model_rose = torch.load('0119_AngeL_rose_model_4expert_seq128_batchsize64_MIXED_WP3.pth')
    # model_rose = torch.load('0119_MoMoE_batch64_seq128_4t16e_MIXED_WP3.pth')



    model.load_state_dict(model_rose.state_dict())

    

    # router.bert.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    # for i in range(config.num_hidden_layers):
    #     router.bert.layers.layers[i].load_state_dict(model0.bert.encoders.layers[i].state_dict())

    # router.head.load_state_dict(model0.head.state_dict())

    train_loader1, val_loader = dataset1.train_loader, dataset1.val_loader
    train_loader2, val_loader = dataset2.train_loader, dataset1.val_loader
    train_loader3, val_loader = dataset3.train_loader, dataset1.val_loader

    # print(train_loader1)
    # train_loader2 = dataset2.train_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()


    
    num_updates = num_epochs * len(train_loader1)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    # model, optimizer, lr_scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    model, optimizer, lr_scheduler, train_loader1= accelerator.prepare(model, optimizer, lr_scheduler, train_loader1)
    model.to(device)
    device2 = torch.device("cuda:1")
    # model_baseline.to(device2)
    # model_baseline_moe.to(device2)

    # FNNP1 = sum([p.abs().mean() for p in model.layers.layers[1].experts[0].parameters()])
    # FNNP2 = sum([p.abs().mean() for p in model.layers.layers[1].experts[1].parameters()])


    # print(FNNP1,FNNP2)

    # if freze_emb:
    #     for para in model.embeddings.parameters():
    #         para.requires_grad = False
    
    
        
    model.eval()
    ATT_OUTS = [[]for i in range(12)]
    FFN_OUTS = [[]for i in range(12)]
    OUTPUTS = [[]for i in range(12)]
    ATT_SELF = [[]for i in range(12)]
    CLUSTER1 = [[]for i in range(12)]
    CLUSTER2 = [[]for i in range(12)]

    for i, batch in enumerate(train_loader1):
        
        # hidden_states_for_router = []
        batch = {key: tensor.to(device2) for key, tensor in batch.items()}
        
        # _,_,outputs,att_outs,ffn_outputs,att_self = model_baseline(**batch)
        _,_,att_self,outputs = model_baseline_moe(**batch)
        print(outputs[2].shape)
        for j in range(12):
            # ATT_OUTS[j].append(att_outs[j])
            # FFN_OUTS[j].append(ffn_outputs[j])
            # OUTPUTS[j].append(outputs[j])
            # ATT_SELF[j].append(att_self[j])
            CLUSTER1[j].append(att_self[j])
            CLUSTER2[j].append(outputs[j])



    for j in range(12):
        # ATT_OUTS[j]=torch.cat(ATT_OUTS[j]).view(-1,768).mean(0)
        # FFN_OUTS[j]=torch.cat(FFN_OUTS[j]).view(-1,768).mean(0)
        # OUTPUTS[j]=torch.cat(OUTPUTS[j]).view(-1,768).mean(0)
        CLUSTER1[j]=torch.cat(CLUSTER1[j])
        CLUSTER2[j]=torch.cat(CLUSTER2[j])





    for j in range(12):
        s = "ATT_SELF"
        plt.figure(j+400)
        print(CLUSTER1[j].shape)
        A = CLUSTER1[j].mean(1).cpu().detach().numpy()
        U = umap.UMAP(random_state=42).fit(A)
        UU = U.transform(A)

        plt.scatter(UU[:, 0], UU[:, 1], s=5)

        plt.legend()
        plt.grid(True)
        plt.savefig("0120-%s/CLUSTER-%s%dlayer.png"%(m1,s,j))
    
    for j in range(12):
        s = "OUTPUTS"
        plt.figure(j+300)
        print(CLUSTER2[j].shape)
        A = CLUSTER2[j].mean(1).cpu().detach().numpy()

        U = umap.UMAP(random_state=42).fit(A)
        UU = U.transform(A)

        plt.scatter(UU[:, 0], UU[:, 1], s=5)

        plt.legend()
        plt.grid(True)
        plt.savefig("0120-%s/CLUSTER-%s%dlayer.png"%(m1,s,j))

    # for j in range(12):
    #     s = "ATT_SELF"
    #     plt.figure(j+900)
    #     print(CLUSTER1[j].shape)
    #     A = CLUSTER1[j].view(-1,CLUSTER1[j].shape[-1]).cpu().detach().numpy()
    #     U = umap.UMAP(random_state=42).fit(A)
    #     UU = U.transform(A)

    #     plt.scatter(UU[:, 0], UU[:, 1], s=5)

    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig("0120-%s/tokensCLUSTER-%s%dlayer.png"%(m1,s,j))
    
    # for j in range(12):
    #     s = "OUTPUTS"
    #     plt.figure(j+1000)
    #     print(CLUSTER2[j].shape)
    #     A = CLUSTER2[j].view(-1,CLUSTER1[j].shape[-1]).cpu().detach().numpy()

    #     U = umap.UMAP(random_state=42).fit(A)
    #     UU = U.transform(A)

    #     plt.scatter(UU[:, 0], UU[:, 1], s=5)

    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig("0120-%s/tokensCLUSTER-%s%dlayer.png"%(m1,s,j))
    


    # ATT_OUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # FFN_OUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # OUTPUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # ATT_SELF = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    
    # CLUSTER1 = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # CLUSTER2 = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # CLUSTER3 = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    # CLUSTER4 = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    
    # # t1 = [[0 for i in range(2)] for j in range(12)]
    # # ORTH = []
    # for i, batch in enumerate(train_loader1):

    #     print(i)
    #     # hidden_states_for_router = []
    #     batch = {key: tensor.to(device) for key, tensor in batch.items()}
    #     # print(epoch, i)
    #     # print(next(model.parameters()).device)
    #     # for key, tensor in batch.items():
    #     #     print(f"{key} is on {tensor.device}")
      
        
    #     _, _, _, outputs, _,att_outs,ffn_outputs,att_self = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers)
    #     # print(len(att_self))
    #     # print(len(att_self[0][0]))
    #     # print(att_self[6][0].shape)
    #     for j in range(12):
    #         for e in range(config.num_transformer):

    #             if len(att_self[j][e]):
    #                 print("layer",j,"expert",e)

    #                 CLUSTER1[j][e].append(att_self[j][e])
    #                 CLUSTER2[j][e].append(outputs[j][e])
    # for i, batch in enumerate(train_loader2):

    #     print(i)
    #     # hidden_states_for_router = []
    #     batch = {key: tensor.to(device) for key, tensor in batch.items()}
    #     # print(epoch, i)
    #     # print(next(model.parameters()).device)
    #     # for key, tensor in batch.items():
    #     #     print(f"{key} is on {tensor.device}")
      
        
    #     _, _, _, outputs, _,att_outs,ffn_outputs,att_self = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers)
    #     # print(len(att_self))
    #     # print(len(att_self[0][0]))
    #     # print(att_self[6][0].shape)
    #     for j in range(12):
    #         for e in range(config.num_transformer):

    #             if len(att_self[j][e]):
    #                 print("layer",j,"expert",e)

    #                 # CLUSTER1[j][e].append(att_self[j][e])
    #                 CLUSTER3[j][e].append(outputs[j][e])

    # for i, batch in enumerate(train_loader3):

    #     print(i)
    #     # hidden_states_for_router = []
    #     batch = {key: tensor.to(device) for key, tensor in batch.items()}
    #     # print(epoch, i)
    #     # print(next(model.parameters()).device)
    #     # for key, tensor in batch.items():
    #     #     print(f"{key} is on {tensor.device}")
      
        
    #     _, _, _, outputs, _,att_outs,ffn_outputs,att_self = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers)
    #     # print(len(att_self))
    #     # print(len(att_self[0][0]))
    #     # print(att_self[6][0].shape)
    #     for j in range(12):
    #         for e in range(config.num_transformer):

    #             if len(att_self[j][e]):
    #                 print("layer",j,"expert",e)

    #                 # CLUSTER1[j][e].append(att_self[j][e])
    #                 CLUSTER4[j][e].append(outputs[j][e])
    
    # for j in range(12):
    #     for e in range(config.num_transformer):
    #         # ATT_OUTS[j][e]=torch.cat(ATT_OUTS[j][e]).view(-1,768).mean(0)
    #         # FFN_OUTS[j]=torch.cat(FFN_OUTS[j]).view(-1,768).mean(0)
    #         # OUTPUTS[j]=torch.cat(OUTPUTS[j]).view(-1,768).mean(0)
    #         # ATT_SELF[j][e]=torch.cat(ATT_SELF[j][e])
    #         if len(CLUSTER1[j][e]):
    #             CLUSTER1[j][e]=torch.cat(CLUSTER1[j][e])
    #             CLUSTER2[j][e]=torch.cat(CLUSTER2[j][e])
        
    # for j in range(12):
    #     for e in range(config.num_transformer):
    #         # ATT_OUTS[j][e]=torch.cat(ATT_OUTS[j][e]).view(-1,768).mean(0)
    #         # FFN_OUTS[j]=torch.cat(FFN_OUTS[j]).view(-1,768).mean(0)
    #         # OUTPUTS[j]=torch.cat(OUTPUTS[j]).view(-1,768).mean(0)
    #         # ATT_SELF[j][e]=torch.cat(ATT_SELF[j][e])
    #         if len(CLUSTER3[j][e]):
    #             CLUSTER3[j][e]=torch.cat(CLUSTER3[j][e])
    #             # CLUSTER2[j][e]=torch.cat(CLUSTER2[j][e])
    # for j in range(12):
    #     for e in range(config.num_transformer):
    #         # ATT_OUTS[j][e]=torch.cat(ATT_OUTS[j][e]).view(-1,768).mean(0)
    #         # FFN_OUTS[j]=torch.cat(FFN_OUTS[j]).view(-1,768).mean(0)
    #         # OUTPUTS[j]=torch.cat(OUTPUTS[j]).view(-1,768).mean(0)
    #         # ATT_SELF[j][e]=torch.cat(ATT_SELF[j][e])
    #         if len(CLUSTER4[j][e]):
    #             CLUSTER4[j][e]=torch.cat(CLUSTER4[j][e])
    
    # # for j in range(12):
    # #     s = "ATT_SELF"
    # #     plt.figure(j+100)

    # #     A = torch.cat([CLUSTER1[j][e] for e in range(config.num_transformer) if len(CLUSTER1[j][e])],dim=0).mean(1).cpu().detach().numpy()
    # #     U = umap.UMAP(random_state=42).fit(A)
    # #     UU = U.transform(A)
    # #     for e in range(config.num_transformer):
    # #         if len(CLUSTER1[j][e]):
    # #             A1 = CLUSTER1[j][e].mean(1).cpu().detach().numpy()
    # #             U1 = U.transform(A1)
    # #             plt.scatter(U1[:, 0], U1[:, 1],label=e, s=5)
        
    # #     plt.legend()
    # #     plt.grid(True)
    # #     plt.savefig("0120-%s/CLUSTER-%s%dlayer.png"%(m2,s,j))

    # #     plt.figure(j+500)
    # #     plt.scatter(UU[:, 0], UU[:, 1],  s=5)

    # #     plt.savefig("0120-%s/ALL-CLUSTER-%s%dlayer.png"%(m2,s,j))

    # for j in range(12):
    #     s = "OUTPUTS"
    #     plt.figure(j+200)

    #     AT = torch.cat([CLUSTER2[j][e] for e in range(config.num_transformer) if len(CLUSTER2[j][e])],dim=0).mean(1)
    #     AL = torch.cat([CLUSTER3[j][e] for e in range(config.num_transformer) if len(CLUSTER3[j][e])],dim=0).mean(1)
    #     AR = torch.cat([CLUSTER4[j][e] for e in range(config.num_transformer) if len(CLUSTER4[j][e])],dim=0).mean(1)
    #     # print(j,AT.shape,AL.shape,AR.shape)
    #     A = torch.cat((AT,AL,AR)).cpu().detach().numpy()
    #     U = umap.UMAP(random_state=42).fit(A)
    #     UU = U.transform(A)
    #     for e in range(config.num_transformer):
    #         if len(CLUSTER2[j][e]):
    #             A1 = CLUSTER2[j][e].mean(1).cpu().detach().numpy()
    #             U1 = U.transform(A1)
    #             plt.scatter(U1[:, 0], U1[:, 1],label=str(e)+"W",s=5)

    #         if len(CLUSTER3[j][e]):
    #             A1 = CLUSTER3[j][e].mean(1).cpu().detach().numpy()
    #             U1 = U.transform(A1)
    #             plt.scatter(U1[:, 0], U1[:, 1],label=str(e)+'L',s=5)
    #         if len(CLUSTER4[j][e]):
    #             A1 = CLUSTER4[j][e].mean(1).cpu().detach().numpy()
    #             U1 = U.transform(A1)
    #             plt.scatter(U1[:, 0], U1[:, 1],label=str(e)+'R', s=5)
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig("0120-%s/CLUSTER-%s%dlayer.png"%(m2,s,j))

    #     plt.figure(j+600)
    #     plt.scatter(UU[:, 0], UU[:, 1],  s=5)

    #     plt.savefig("0120-%s/ALL-CLUSTER-%s%dlayer.png"%(m2,s,j))


    # # for j in range(12):
    # #     s = "ATT_SELF"
    # #     plt.figure(j+3300)

    # #     A = torch.cat([CLUSTER1[j][e] for e in range(config.num_transformer) if len(CLUSTER1[j][e])],dim=0).view(-1,config.hidden_size).cpu().detach().numpy()
    # #     U = umap.UMAP(random_state=42).fit(A)
    # #     UU = U.transform(A)
    # #     for e in range(config.num_transformer):
    # #         if len(CLUSTER1[j][e]):
    # #             A1 = CLUSTER1[j][e].mean(1).cpu().detach().numpy()
    # #             U1 = U.transform(A1)
    # #             plt.scatter(U1[:, 0], U1[:, 1],label=e, s=5)
        
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig("0120-%s/tokensCLUSTER-%s%dlayer.png"%(m2,s,j))

    #     # plt.figure(j+500)
    #     # plt.scatter(UU[:, 0], UU[:, 1],  s=5)

    #     # plt.savefig("0120-%s/ALL-CLUSTER-%s%dlayer.png"%(m2,s,j))

    # for j in range(12):
    #     s = "OUTPUTS"
    #     plt.figure(j+3400)

    #     A = torch.cat([CLUSTER2[j][e] for e in range(config.num_transformer) if len(CLUSTER1[j][e])],dim=0).view(-1,config.hidden_size).cpu().detach().numpy()
    #     U = umap.UMAP(random_state=42).fit(A)
    #     UU = U.transform(A)
    #     for e in range(config.num_transformer):
    #         if len(CLUSTER2[j][e]):
    #             A1 = CLUSTER2[j][e].mean(1).cpu().detach().numpy()
    #             U1 = U.transform(A1)
    #             plt.scatter(U1[:, 0], U1[:, 1],label=e, s=5)
        
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig("0120-%s/tokensCLUSTER-%s%dlayer.png"%(m2,s,j))

        # plt.figure(j+600)
        # plt.scatter(UU[:, 0], UU[:, 1],  s=5)

        # plt.savefig("0120-%s/ALL-CLUSTER-%s%dlayer.png"%(m2,s,j))
    

    


    

if __name__ == "__main__":
    set_seed(45)
    config0 = BertConfig.from_json_file('config/MoMoE.json')
    config = BertConfig.from_json_file('config/MoMoE.json')
    # dataset = RestaurantForLM_small(config=config)
    dataset1 = MoMoE_MIXED_WIKI103_0124_W(config = config)
    dataset2 = MoMoE_MIXED_WIKI103_0124_L(config = config)
    dataset3 = MoMoE_MIXED_WIKI103_0124_R(config = config)

    torch.cuda.set_device(2)
    device = torch.device("cuda")
    # model = base_models.AngeLB_MoE(config=config)
    model = base_models.AngeL_rose_model(config=config0)
    # model = base_models.MoMoE(config=config0)

    # model = base_models.BertForMLM(config=config)

    router = base_models.BertWithSavers(config=config0)
    router.to(device)
    model.to(device)
    # model = nn.DataParallel(model)

    
    show(model=model, num_epochs=50, dataset1=dataset1, device=device, dataset2 = dataset2, dataset3=dataset3,router = router)