import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import Wikitxt103ForLM_80W,MixedData_1121,MoMoE_MIXED_WIKI103_0124_3,MoMoE_MIXED_WIKI103_0124_1,MoMoE_MIXED_WIKI103_0124_2,MoMoE_MIXED_TOSHOW,MoMoE_WIKI103_TOSHOW,MixedData_1211,Wikitxt103ForLM_0109,RestaurantforLM_0109,ACLForLM_0109,RestaurantForLM_small, MixedData,MixedData_stage1,Mixdata_1103,Mixdata_1115,Wikitxt103ForLM_1103,Wikitxt103ForLM_0102_rose
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

from sklearn.cluster import KMeans



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
        output,ATT_SELF,O,inputs = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores,ATT_SELF,O,inputs

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

def show(model, num_epochs, dataset1, device, dataset2,dataset3):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    




    MOE = 0
    BERT = 0
    MOMOE_SHAREDW = 1
    MOMOE_GDS = 0
    device2 = torch.device("cuda:7")
    device3 = torch.device("cuda:6")
    if MOE:
        model_baseline_moe = BertForMLM(config = BertConfig.from_json_file('config/MoMoE.json'))  


        model_baseline0_moe =torch.load('0308-moe-64*128-FEWER_SPECIFIC_GENERAL-4E-lr3e-4.pth')

        model_baseline_moe.load_state_dict(model_baseline0_moe.state_dict())
        model_baseline_moe.to(device3)

        model_baseline_moe.eval()
    if BERT:
        
        model_baseline = base_models.BertForMLM_toshow(config=config0)


        model_baseline0 =torch.load('0311-BERT-MIXED_LEGAL_REVIEW.pth')

        model_baseline.load_state_dict(model_baseline0.state_dict())

        model_baseline.to(device3)

        model_baseline.eval()


    PRO_VECS = []
    for l in range(config.num_hidden_layers):
        eig_vecs0 = torch.load('0229-layer%d_pro_vec-2t-MIXED_LEGAL_PUBMED.pth'%l, map_location='cuda')
        PRO_VECS.append(eig_vecs0)



    train_loader1, val_loader = dataset1.train_loader, dataset1.val_loader
    train_loader2, val_loader = dataset2.train_loader, dataset1.val_loader
    train_loader3, val_loader = dataset3.train_loader, dataset1.val_loader

    # print(train_loader1)
    # train_loader2 = dataset2.train_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    model0 = base_models.BertForMLM_toshow(config=config0)
    model01 = torch.load('/home/jxzhou/PLM_PER/BERT-CL-main/0226-bert-batch64_seq128-MIXED_LEGAL_PUBMED-pretrain-lr2e-4-for-routing2.pth')
    model0.load_state_dict(model01.state_dict())
    # model = torch.load('0226_MoMoE_uniqueatt_commonffn_batch64_seq128_2t4e_MIXED_LEGAL_PUBMED_autoinit_WP128000-pretrain-lr2e-4-correct2-80w*5-commonunique.pth')
    model = torch.load('0307_MoMoE_uniqueatt_commonffn_batch64_seq128_2t4e_MIXED_LEGAL_PUBMED-pretrain-lr3e-4-correct2-160w*3-commonunique.pth')
    
    cluster_centers = load_layer_data('/home/jxzhou/PLM_PER/BERT-CL-main/0226-layer_centers-2t-MIXED_LEGAL_PUBMED2.pth',device)
    # model = torch.load('/home/jxzhou/PLM_PER/BERT-CL-main/0308_MoMoE_uniqueatt_commonffn_batch64_seq128_2t4e_FEWER_SPECIFIC_GENERAL-pretrain-lr4e-4-correct2-160w*3-GDS.pth')
    # cluster_centers = load_layer_data('0229-layer_centers-2t-MIXED_LEGAL_PUBMED2-unique.pth',device)
    
    num_updates = num_epochs * len(train_loader1)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    # model, optimizer, lr_scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    model, optimizer, lr_scheduler, train_loader1,train_loader2= accelerator.prepare(model, optimizer, lr_scheduler, train_loader1,train_loader2)
    model.to(device)
    
    model0.to(device2)


    
    
        
    model.eval()


    if MOE or BERT:
        inputs1 = [[]for j in range(config.num_hidden_layers)] 
        inputs2 = [[]for j in range(config.num_hidden_layers)]   
        inputs3 = [[]for j in range(config.num_hidden_layers)]   

        for i, batch in enumerate(train_loader1):
            batch = {key: tensor.to(device3) for key, tensor in batch.items()}

            print(i)
            _, _ ,ou,_,_= model_baseline_moe(**batch)
            # _, _ ,_,ou,_= model_baseline_moe(**batch)

            # _, _,ou,_,_,_,_ = model_baseline(**batch)

            for j in range(12):
                inputs1[j].append(ou[j].detach())
                    
                        
        for i, batch in enumerate(train_loader2):
            batch = {key: tensor.to(device3) for key, tensor in batch.items()}
            print(i)
            _, _ ,ou,_,_= model_baseline_moe(**batch)
            # _, _ ,_,ou,_= model_baseline_moe(**batch)
            # _, _,ou,_,_,_,_ = model_baseline(**batch)


            for j in range(12):
                inputs2[j].append(ou[j].detach())
        for i, batch in enumerate(train_loader3):
            batch = {key: tensor.to(device3) for key, tensor in batch.items()}
            print(i)
            _, _ ,ou,_,_= model_baseline_moe(**batch)
            # _, _ ,_,ou,_= model_baseline_moe(**batch)
            # _, _,ou,_,_,_,_ = model_baseline(**batch)


            for j in range(12):
                inputs3[j].append(ou[j].detach())

        for j in range(12):
            inputs1[j] = torch.cat(inputs1[j])
            inputs2[j] = torch.cat(inputs2[j])
            inputs3[j] = torch.cat(inputs3[j])
            # inputs1[j]=torch.matmul(inputs1[j],PRO_VECS[j])[:,:,:-config.unique_hidden_size]

            # inputs2[j]=torch.matmul(inputs2[j],PRO_VECS[j])[:,:,:-config.unique_hidden_size]

            # inputs3[j]=torch.matmul(inputs3[j],PRO_VECS[j])[:,:,:-config.unique_hidden_size]

            
        for j in range(12):
            s = "uniqueatt"
            plt.figure(j+200)


            # A = torch.cat((inputs1[j],inputs2[j],inputs3[j])).mean(1).cpu().detach().numpy()
            A = inputs1[j].mean(1).cpu().detach().numpy()

            U = umap.UMAP(random_state=42).fit(A)
            
                
            A1 = inputs1[j].mean(1).cpu().detach().numpy()
            U1 = U.transform(A1)
            plt.scatter(U1[:, 0], U1[:, 1],label="Legal",s=5)

            
            # A2 = inputs2[j].mean(1).cpu().detach().numpy()
            # U2 = U.transform(A2)
            # plt.scatter(U2[:, 0], U2[:, 1],label='PubMed',s=5)

            # A3 = inputs3[j].mean(1).cpu().detach().numpy()
            # U3 = U.transform(A3)
            # plt.scatter(U3[:, 0], U3[:, 1],label='WIKI',s=5)

            plt.legend()
            plt.grid(True)
            plt.xlim(-20,20)
            plt.ylim(-20,20)
            # plt.savefig("0313-BERT/BERTCLUSTER-%s%dlayer.png"%(s,j))
            plt.savefig("0313-MoE/MoECLUSTER-%s%dlayer.png"%(s,j))

    


    ATT_OUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    FFN_OUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    OUTPUTS = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    ATT_SELF = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    
    CLUSTER1 = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    CLUSTER2 = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    CLUSTER3 = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    CLUSTER4 = [[[]for j in range(config.num_experts)]for i in range(config.num_hidden_layers)]
    

    concated_outputs1 = [[]for j in range(config.num_hidden_layers)]
    concated_outputs2 = [[]for j in range(config.num_hidden_layers)]
    concated_outputs3 = [[]for j in range(config.num_hidden_layers)]

    # t1 = [[0 for i in range(2)] for j in range(12)]
    # ORTH = []



    for i, batch in enumerate(train_loader1):

        print(i)

        batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
        batch = {key: tensor.to(device) for key, tensor in batch.items()}

        
        
        _,_,_,_,_,_,inputs = model0(**batch0)

        # inputs = [i0.to(device) for i0 in inputs]
        inputs = [inputs[-1].to(device) for i in range(12)]

        cluster_centers = [cluster_centers[-1].to(device) for i in range(12)]
    

        _, _, _, outputs, _,_,_,_,cat_out,common_out = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers,inputs,PRO_VECS)

        for j in range(12):
            # for e in range(config.num_transformer):

            #     if len(outputs[j][e]):
            #         print("layer",j,"transformer",e)

            #         # CLUSTER1[j][e].append(att_self[j][e])
            #         CLUSTER2[j][e].append(outputs[j][e].detach())
            concated_outputs1[j].append(cat_out[j].detach())
            concated_outputs2[j].append(outputs[j].detach())
    # for i, batch in enumerate(train_loader2):

    #     print(i)
    #     batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
    #     batch = {key: tensor.to(device) for key, tensor in batch.items()}

        
        
    #     _,_,_,_,_,_,inputs = model0(**batch0)

    #     # inputs = [i0.to(device) for i0 in inputs]
    #     inputs = [inputs[-1].to(device) for i in range(12)]

    #     cluster_centers = [cluster_centers[-1].to(device) for i in range(12)]


    #     _, _, _, _, _,_,_,_,outputs  = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers,inputs,PRO_VECS)

        
        
    #     for j in range(12):
    #         concated_outputs2[j].append(outputs[j].detach())
    #         # for e in range(config.num_transformer):
                
    #         #     if len(outputs[j][e]):
    #         #         print("layer",j,"transformer",e)

    #         #         # CLUSTER1[j][e].append(att_self[j][e])
    #         #         CLUSTER3[j][e].append(outputs[j][e].detach())

    # for i, batch in enumerate(train_loader3):

    #     print(i)
    #     batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
    #     batch = {key: tensor.to(device) for key, tensor in batch.items()}

        
        
    #     _,_,_,_,_,_,inputs = model0(**batch0)

    #     inputs = [i0.to(device) for i0 in inputs]

    #     _, _, _, _, _,_,_,_,outputs  = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers,inputs,PRO_VECS)

        
        
    #     for j in range(12):
    #         concated_outputs3[j].append(outputs[j].detach())
    #         # for e in range(config.num_transformer):
                
    #             # if len(outputs[j][e]):
    #             #     print("layer",j,"transformer",e)

    #             #     # CLUSTER1[j][e].append(att_self[j][e])
    #             #     CLUSTER3[j][e].append(outputs[j][e])

    
    for j in range(12):
        concated_outputs1[j]=torch.cat(concated_outputs1[j])
        # concated_outputs1[j]=torch.matmul(concated_outputs1[j],PRO_VECS[j])[:,:,:-config.unique_hidden_size]
        # for e in range(config.num_transformer):

        #     if len(CLUSTER2[j][e]):
        #         # CLUSTER1[j][e]=torch.cat(CLUSTER1[j][e])
        #         CLUSTER2[j][e]=torch.cat(CLUSTER2[j][e])
        
    for j in range(12):
        concated_outputs2[j]=torch.cat(concated_outputs2[j])
        # concated_outputs2[j]=torch.matmul(concated_outputs2[j],PRO_VECS[j])[:,:,:-config.unique_hidden_size]

        # for e in range(config.num_transformer):

        #     if len(CLUSTER3[j][e]):
        #         CLUSTER3[j][e]=torch.cat(CLUSTER3[j][e])
        #         # CLUSTER2[j][e]=torch.cat(CLUSTER2[j][e])

    # for j in range(12):
    #     concated_outputs3[j]=torch.cat(concated_outputs3[j])
    #     # concated_outputs3[j]=torch.matmul(concated_outputs3[j],PRO_VECS[j])[:,:,:-config.unique_hidden_size]

    #     # for e in range(config.num_transformer):

    #         # if len(CLUSTER3[j][e]):
    #         #     CLUSTER3[j][e]=torch.cat(CLUSTER3[j][e])
    #         #     # CLUSTER2[j][e]=torch.cat(CLUSTER2[j][e])
    # for j in range(12):
    #     s = "INPUTS"
    #     plt.figure(j+200)

    #     AT = torch.cat([CLUSTER2[j][e] for e in range(config.num_transformer) if len(CLUSTER2[j][e])],dim=0).mean(1)
    #     AL = torch.cat([CLUSTER3[j][e] for e in range(config.num_transformer) if len(CLUSTER3[j][e])],dim=0).mean(1)
    #     # AR = torch.cat([CLUSTER4[j][e] for e in range(config.num_transformer) if len(CLUSTER4[j][e])],dim=0).mean(1)
    #     # print(j,AT.shape,AL.shape,AR.shape)
    #     A = torch.cat((AT,AL)).cpu().detach().numpy()
    #     U = umap.UMAP(random_state=42).fit(A)
    #     UU = U.transform(A)
    #     for e in range(config.num_transformer):
    #         if len(CLUSTER2[j][e]):
    #             A1 = CLUSTER2[j][e].mean(1).cpu().detach().numpy()
    #             U1 = U.transform(A1)
    #             plt.scatter(U1[:, 0], U1[:, 1],label=str(e)+"Legal",s=5)

    #         if len(CLUSTER3[j][e]):
    #             A1 = CLUSTER3[j][e].mean(1).cpu().detach().numpy()
    #             U1 = U.transform(A1)
    #             plt.scatter(U1[:, 0], U1[:, 1],label=str(e)+'Review',s=5)
    #         # if len(CLUSTER4[j][e]):
    #         #     A1 = CLUSTER4[j][e].mean(1).cpu().detach().numpy()
    #         #     U1 = U.transform(A1)
    #         #     plt.scatter(U1[:, 0], U1[:, 1],label=str(e)+'R', s=5)
    #     plt.legend()
    #     plt.grid(True)
    #     plt.xlim(-20,20)
    #     plt.ylim(-20,20)
    #     plt.savefig("0304-MoMoEshared/CLUSTER-%s%dlayer.png"%(s,j))

    #     plt.figure(j+600)
    #     plt.scatter(UU[:, 0], UU[:, 1],  s=5)

    #     plt.savefig("0304-MoMoEshared/ALL-CLUSTER-%s%dlayer.png"%(s,j))

    for j in range(12):
        s = "full_dimension_catatt_and_outputs"
        plt.figure(j+200)


        # A = torch.cat((concated_outputs1[j],concated_outputs2[j])).mean(1).cpu().detach().numpy()
        A = concated_outputs1[j].mean(1).cpu().detach().numpy()
        
        U = umap.UMAP(random_state=42).fit(A)
        
            
        A1 = concated_outputs1[j].mean(1).cpu().detach().numpy()
        U1 = U.transform(A1)
        plt.scatter(U1[:, 0], U1[:, 1],label="cat_att",s=5)


        
        A2 = concated_outputs2[j].mean(1).cpu().detach().numpy()
        U2 = U.transform(A2)
        plt.scatter(U2[:, 0], U2[:, 1],label='common_att',s=5, alpha = 0.1)

        # A3 = concated_outputs3[j].mean(1).cpu().detach().numpy()
        # U3 = U.transform(A3)
        # plt.scatter(U3[:, 0], U3[:, 1],label='WIKI',s=5)

        plt.legend()
        plt.grid(True)
        plt.xlim(-10,40)
        plt.ylim(-10,40)
        plt.savefig("0304-MoMoEshared/CLUSTER-%s%dlayer.png"%(s,j))

    # for j in range(12):
    #     s = "OUTPUTS"
    #     plt.figure(j+400)


        
            
    #     A1 = concated_outputs1[j].view(-1,768).cpu().mean(0).detach().numpy()
    #     plt.scatter(A1,label="Legal",s=5)


        
    #     A2 = concated_outputs2[j].view(-1,768).mean(0).cpu().detach().numpy()
    #     plt.scatter(A2,label='PubMed',s=5)

    #     plt.legend()
    #     plt.grid(True)
    #     plt.xlim(-20,20)
    #     plt.ylim(-20,20)
    #     plt.savefig("0304-MoMoEGDS/CLUSTER-%s%dlayer.png"%(s,j))
    
    # for j in range(11,12):
    #     plt.figure(j+200)
    #     s = 'fingegrained'
    #     A = inputs3[j].mean(1).cpu().detach().numpy()
    #     U = umap.UMAP(random_state=42).fit(A)


    #     A3 = inputs3[j].mean(1).cpu().detach().numpy()
    #     U3 = U.transform(A3)


       
    #     kmeans = KMeans(n_clusters=8, random_state=42)
    #     kmeans.fit(U3)

    #     # 获取聚类标签
    #     labels = kmeans.labels_
    #     plt.scatter(U3[:, 0], U3[:, 1],c=labels,s=5)


    #     plt.legend()
    #     plt.grid(True)
    #     plt.xlim(0,20)
    #     plt.ylim(0,20)
    #     plt.savefig("0304-MoMoEshared/MoECLUSTER-%s%dlayer.png"%(s,j))

    #     # # 将原始数据的索引标号和聚类标签整理成字典
    #     # cluster_dict = {index: label for index, label in enumerate(labels)}


    #     plt.figure(j+300)


    #     A = concated_outputs3[j].mean(1).cpu().detach().numpy()
    #     U = umap.UMAP(random_state=42).fit(A)
        
            

    #     A3 = concated_outputs3[j].mean(1).cpu().detach().numpy()
    #     U3 = U.transform(A3)
    #     plt.scatter(U3[:, 0], U3[:, 1],c=labels,s=5)

    #     plt.legend()
    #     plt.grid(True)
    #     plt.xlim(0,20)
    #     plt.ylim(0,20)
    #     plt.savefig("0304-MoMoEshared/CLUSTER-%s%dlayer.png"%(s,j))
           


    


    

if __name__ == "__main__":
    set_seed(45)
    config0 = BertConfig.from_json_file('config/bert.json')
    config = BertConfig.from_json_file('config/MoMoE.json')
    # dataset = RestaurantForLM_small(config=config)
    dataset1 = MoMoE_MIXED_WIKI103_0124_1(config = config)
    dataset2 = MoMoE_MIXED_WIKI103_0124_2(config = config)
    dataset3 = MoMoE_MIXED_WIKI103_0124_3(config = config)

    torch.cuda.set_device(6)
    device = torch.device("cuda")

    # model = base_models.MoMoE_0306_narrowneck(config=config)
    model = base_models.MoMoE_0126(config=config)


    model.to(device)
    # model = nn.DataParallel(model)

    
    show(model=model, num_epochs=50, dataset1=dataset1, device=device, dataset2 = dataset2, dataset3=dataset3)