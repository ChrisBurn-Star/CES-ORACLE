import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForMaskedLM
from transformer.Transformer import TransformerEncoder,MoMoE_layer_0315,BertModel,Embeddings,MoMoE_layer_uniqueatt_commonffn,MoMoE_layer_0229,MoMoE_layer_0226
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertOnlyNSPHead
from transformers import BertConfig, get_cosine_schedule_with_warmup
from accelerate import Accelerator,load_checkpoint_and_dispatch
from accelerate import DistributedDataParallelKwargs as DDPK
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from Dataset import GLUE
from Dataset_new import GAD,Overruling,GAD_single,medical_abstract,multi_domains,Casehold,EUADR,SST2,COLA,MRPC,QQP,RTE,QNLI
import util
import json
import base_models
import numpy as np
import pandas as pd
import random
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    return layer_data
def get_long_data(tensor):
    not_ones = tensor != 1

    # 计算不为1的元素个数
    count_not_ones = torch.sum(not_ones)

    

    return count_not_ones

class MoMoEForCLS(nn.Module):
    def __init__(self, config) -> None:
        super(MoMoEForCLS, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_0226(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.pooler = BertPooler(config)
        self.head = nn.Linear(config.hidden_size, 5)
        self.criterion = nn.CrossEntropyLoss()
        self.deval = 1
    
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
    
    def forward(self, input_ids, attention_mask, label, cluster_centers,hi_ou,pro_vecs):
        hidden_states = self.embeddings(input_ids)
        # inputs = []
        # outputs = []
        # att_outs = []
        # ffn_outs = []
        # att_self = []
        expert_ids = []
        # inputs.append(hidden_states)
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i],hi_ou[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,cat_att,_ = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hi_ou[i])
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            # att_outs.append(att_outputs)
            # outputs.append(inputs)
            # ffn_outs.append(ffn_outputs)
            # att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
            # inputs.append(hidden_states)

        pooled_output = self.pooler(hidden_states)
        score = self.head(pooled_output)
        cls_loss = self.criterion(score, label)

        return cls_loss, score,expert_ids



class MoMoEForCLS_every(nn.Module):
    def __init__(self, config) -> None:
        super(MoMoEForCLS_every, self).__init__()
        self.config = config
        self.num_momoelayers = 2
        self.bert_layers = nn.ModuleList([TransformerEncoder(config) for i in range(config.num_hidden_layers-self.num_momoelayers)])
        self.layers = nn.ModuleList([MoMoE_layer_0315(config) for i in range(self.num_momoelayers)])
        self.embeddings = Embeddings(config)
        self.pooler = BertPooler(config)
        self.head = nn.Linear(config.hidden_size, 5)
        self.criterion = nn.CrossEntropyLoss()
        self.deval = 1
    
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
    
    def forward(self, input_ids, attention_mask, label, cluster_centers,hi_ou,pro_vecs):
        hidden_states = self.embeddings(input_ids)
        for j in range(self.config.num_hidden_layers-self.num_momoelayers):
            hidden_states = self.bert_layers[j](hidden_states,attention_mask)
        # inputs = []
        # outputs = []
        # att_outs = []
        # ffn_outs = []
        # att_self = []
        expert_ids = []
        # inputs.append(hidden_states)
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[self.config.num_hidden_layers-self.num_momoelayers+i],hi_ou[self.config.num_hidden_layers-self.num_momoelayers+i])

            hidden_states,att_outputs,ffn_outputs,at_out_selfs,input,CONCAT_ATTENTION_OUT = self.layers[i](hidden_states, attention_mask, cluster_centers[self.config.num_hidden_layers-self.num_momoelayers+i],hi_ou[self.config.num_hidden_layers-self.num_momoelayers+i])
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            # att_outs.append(att_outputs)
            # outputs.append(inputs)
            # ffn_outs.append(ffn_outputs)
            # att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
            # inputs.append(hidden_states)

        pooled_output = self.pooler(hidden_states)
        score = self.head(pooled_output)
        cls_loss = self.criterion(score, label)

        return cls_loss, score,expert_ids

class MoMoEForCLS2(nn.Module):
    def __init__(self, config) -> None:
        super(MoMoEForCLS2, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_0226(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.pooler = BertPooler(config)
        self.head = nn.Linear(config.hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.deval = 1
    
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
    
    def forward(self, input_ids, attention_mask, label, cluster_centers,hi_ou,pro_vecs):
        hidden_states = self.embeddings(input_ids)
        # inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        # inputs.append(hidden_states)
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i],hi_ou[i],pro_vecs[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,cat_att,_ = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hi_ou[i],pro_vecs[i])
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            # att_outs.append(att_outputs)
            # outputs.append(inputs)
            # ffn_outs.append(ffn_outputs)
            # att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
            # inputs.append(hidden_states)

        pooled_output = self.pooler(hidden_states)
        score = self.head(pooled_output)
        cls_loss = self.criterion(score, label)

        return cls_loss, score,expert_ids

class BERTForCLS(nn.Module):
    def __init__(self, config) -> None:
        super(BERTForCLS, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.pooler = BertPooler(config)
        self.head = nn.Linear(config.hidden_size, 2)
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
    
    def forward(self, input_ids, attention_mask, label):
        output,_ = self.bert(input_ids, attention_mask)
        pooled_output = self.pooler(output)
        score = self.head(pooled_output)
        cls_loss = self.criterion(score, label)

        return cls_loss, score



def finetune(model:BERTForCLS, dataset: medical_abstract,k,EPOCH,task,CKPTS,lr,seed,LEGAL):
    
    accelerator = Accelerator()
    train_loader = dataset.train_loader
    test_loader = dataset.val_loader
    # val_loader1, val_loader2,val_loader3,val_loader4= dataset.val_loader1,dataset.val_loader2,dataset.val_loader3,dataset.val_loader4
    # val_loader_longtailed = dataset.val_loader_longtailed

    # modelM = torch.load("/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0315/0320-MoMoE_full_dimension-3072ffns-longtaile-pubmed-2e-4-checkpoint40000.pth")
    # modelM = torch.load("0228_MoMoE_uniqueatt_commonffn_batch64_seq128_2t4e_MIXED_LEGAL_PUBMED_autoinit_WP128000-pretrain-lr2e-4-correct2-80w*5-GDS.pth")
    
    # model.embeddings.load_state_dict(modelM.embeddings.state_dict())
    # model.layers.load_state_dict(modelM.layers.state_dict())
    # model.bert_layers.load_state_dict(modelM.bert_layers.state_dict())
    # cluster_centers = load_layer_data('0223_MoMoE_uniqueatt_commonffn_batch64_seq128_2t4e_MIXED_LEGAL_PUBMED_autoinit_WP128000-centers.pth')
    model0 = base_models.BertForMLM(config=config0)
    load_checkpoint_and_dispatch(model0, '/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_REBUTALL/bert-copyparams-med-specific',device_map={"":device})
    model.bert.load_state_dict(model0.bert.state_dict())

    # model0 = torch.load('0119-bert-MIXED-WP1.pth')

    # model0 = torch.load('0119-bert-WIKI103-WP10.pth')
    # cluster_centers2 = load_layer_data('0229-layer_centers-2t-MIXED_LEGAL_PUBMED2-unique.pth')
    # cluster_centers = load_layer_data('0226-layer_centers-2t-MIXED_LEGAL_PUBMED2.pth')
    # for l in range(config.num_hidden_layers):
    #     model.layers[l].state = 1
    #     model.layers[l].cluster_centers = cluster_centers[l]
        # print(model.layers[l].cluster_centers.shape)
    # model0 = torch.load("0223-bert-MIXED_LEGAL_PUBMED-pretrain.pth")
    # model.bert.load_state_dict(model0.bert.state_dict())
    # PRO_VECS = []
    # for l in range(config.num_hidden_layers):
    #     eig_vecs0 = torch.load('0229-layer%d_pro_vec-2t-MIXED_LEGAL_PUBMED.pth'%l, map_location='cuda')
    #     PRO_VECS.append(eig_vecs0)

    writer = SummaryWriter('/home/jxzhou/PLM_PER/BERT-CL-main/rebuttal_finetune/bert-copyparams-med-specific-task%s'%task)
    # writer = SummaryWriter('tensorboard_0223/0301_MoMoE_MIXED_LEGAL_PUBMED_correct2_GDS_finetuned-64*128-GAD_single1-2.5e-5-PR')
    
    
    # writer = SummaryWriter('tensorboard_0223/0225_BERT_MIXED_LEGAL_PUBMED_finetuned-64*128-Overruling')

    optimizer = optim.AdamW([
        {'params': model.parameters()},
        # {'params': model.head.parameters()},
        # {'params': model.pooler.parameters()}
    ], lr=lr, weight_decay=0.02, betas=[0.9, 0.99], eps=1e-6)
    for param in model.bert.parameters():
        param.requires_grad = False
    # model, optimizer, train_loader, test_loader,val_loader1, val_loader2,val_loader3,val_loader4 = accelerator.prepare(model, optimizer, train_loader, test_loader,val_loader1, val_loader2,val_loader3,val_loader4)
    

    # val_loader = [val_loader1, val_loader2,val_loader3,val_loader4]
        
    model, optimizer, train_loader, test_loader= accelerator.prepare(model, optimizer, train_loader, test_loader)

    num_epoch = EPOCH

    for epoch in range(num_epoch):
        
        losses = []

        preds = []
        for i, batch in enumerate(train_loader):
            # print(i)
            # if i == 2:
            #     break
            loss,_ = model(batch['input_ids'],batch['attention_mask'],batch['label'])



            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])

        losses = []
        with torch.no_grad():

            for J, batch in enumerate(test_loader):
   
                loss, score =model(batch['input_ids'],batch['attention_mask'],batch['label'])
                y_pred = torch.argmax(score, dim=-1)
     
                pred = y_pred == batch['label']
 
                losses.append(accelerator.gather(loss.repeat(config.batch_size)))
                preds.append(accelerator.gather(pred))
            loss_test = torch.mean(torch.cat(losses)[:len(test_loader.dataset)])
            preds = torch.cat(preds)
            acc_test = torch.sum(preds) / len(preds)

        # with torch.no_grad():

        #     for J, batch in enumerate(test_loader):
        #         # print(J)
        #         for b in range(batch['input_ids'].shape[0]):
        #             lens.append(get_long_data(batch['input_ids'][b]))
        #             loss,score = model(batch['input_ids'][b].view(1,-1),batch['attention_mask'][b].view(1,-1), batch['label'][b].view(-1))
        #             y_pred = torch.argmax(score, dim=-1)
        #             pred = y_pred == batch['label'][b]

        #             # losses.append(accelerator.gather(loss.repeat(config.batch_size)))
        #             # preds.append(accelerator.gather(pred))
        #             long_acc.append(pred)
        #     # loss_test = torch.mean(torch.cat(losses)[:len(test_loader.dataset)])
        #     # preds = torch.cat(preds)
        #     # acc_test = torch.sum(preds) / len(preds)
        # # print(lens)
        # # print(long_acc)
        # df = pd.DataFrame({
        #     'Column1': [x.item() for x in lens],
        #     'Column2': [x.item() for x in long_acc]
        # })

        # # 保存DataFrame到CSV文件
        # df.to_csv('0428-moe-%s-%d.csv'%(task,epoch), index=True)




        accelerator.print(f'Epoch:{epoch} ({(i + 1) * accelerator.num_processes} Updates), Train Loss: {loss_train}, Test Loss: {loss_test}, Test Acc: {acc_test}')
        if accelerator.is_local_main_process:
            writer.add_scalar('acc_test', acc_test, epoch)
            writer.add_scalar('loss_train', loss_train, epoch)
            writer.add_scalar('loss_val', loss_test, epoch)

            # for i in range(4):
            #     writer.add_scalar('accuracy-dataset%d'%i, ACCS[i], epoch)
            # writer.add_scalar('longtailed-acc_test', acc_test0, epoch)
            # writer.add_scalar('longtailed-P', P_S0, epoch)
            # writer.add_scalar('longtailed-R', R_S0, epoch)
            # writer.add_scalar('longtailed-F1', F10, epoch)



if __name__ == "__main__":
    seed = 14
    set_seed(seed)
    config = BertConfig.from_json_file('config/bert.json')
    config0 = BertConfig.from_json_file('config/bert.json')
    EPOCH = 10
    LEGAL = 110000
    torch.cuda.set_device(6)
    device = torch.device("cuda:6")


    datasets = {}
    datasets['casehold'] = Casehold(config)
    datasets['overruling'] = Overruling(config)

    datasets['sst2'] = SST2(config)

    datasets['gad'] = GAD(config)
    datasets['euadr'] = EUADR(config)




    k = 1

    

    model = BERTForCLS(config)
    # for ck in range(100,1001,100):
    ck = 800
    for i in datasets:
        print(i)
        finetune(model, datasets[i],k,EPOCH,i,ck,1e-3,seed,LEGAL)
    # print(PO/10.0,RO/10.0)