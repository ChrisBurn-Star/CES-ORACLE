import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForMaskedLM
from transformer.Transformer import TransformerEncoder,MoMoE_layer_0315,BertModel,Embeddings,MoMoE_layer_uniqueatt_commonffn,MoMoE_layer_0229,MoMoE_layer_0226
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertOnlyNSPHead
from transformers import BertConfig, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from Dataset import GLUE
from Dataset_new import GAD,Overruling,GAD_single,medical_abstract
import util
import json
import base_models
import numpy as np


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    return layer_data


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
        output = self.bert(input_ids, attention_mask)
        pooled_output = self.pooler(output)
        score = self.head(pooled_output)
        cls_loss = self.criterion(score, label)

        return cls_loss, score



def finetune(model: MoMoEForCLS_every, dataset: medical_abstract,k,EPOCH):
    
    accelerator = Accelerator()
    train_loader = dataset.train_loader
    test_loader = dataset.val_loader
    val_loader_longtailed = dataset.val_loader_longtailed

    modelM = torch.load("/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0315/0320-MoMoE_full_dimension-3072ffns-longtaile-pubmed-2e-4-checkpoint40000.pth")
    # modelM = torch.load("0228_MoMoE_uniqueatt_commonffn_batch64_seq128_2t4e_MIXED_LEGAL_PUBMED_autoinit_WP128000-pretrain-lr2e-4-correct2-80w*5-GDS.pth")
    
    model.embeddings.load_state_dict(modelM.embeddings.state_dict())
    model.layers.load_state_dict(modelM.layers.state_dict())
    # model.bert_layers.load_state_dict(modelM.bert_layers.state_dict())
    # cluster_centers = load_layer_data('0223_MoMoE_uniqueatt_commonffn_batch64_seq128_2t4e_MIXED_LEGAL_PUBMED_autoinit_WP128000-centers.pth')
    model0 = base_models.BertForMLM_toshow(config=config0)
    model01 = torch.load('0226-bert-batch64_seq128-MIXED_LEGAL_PUBMED-pretrain-lr2e-4-for-routing2.pth')
    model0.load_state_dict(model01.state_dict())

    # model0 = torch.load('0119-bert-MIXED-WP1.pth')

    # model0 = torch.load('0119-bert-WIKI103-WP10.pth')
    # cluster_centers2 = load_layer_data('0229-layer_centers-2t-MIXED_LEGAL_PUBMED2-unique.pth')
    cluster_centers = load_layer_data('0226-layer_centers-2t-MIXED_LEGAL_PUBMED2.pth')
    # for l in range(config.num_hidden_layers):
    #     model.layers[l].state = 1
    #     model.layers[l].cluster_centers = cluster_centers[l]
        # print(model.layers[l].cluster_centers.shape)
    # model0 = torch.load("0223-bert-MIXED_LEGAL_PUBMED-pretrain.pth")
    # model.bert.load_state_dict(model0.bert.state_dict())
    PRO_VECS = []
    for l in range(config.num_hidden_layers):
        eig_vecs0 = torch.load('0229-layer%d_pro_vec-2t-MIXED_LEGAL_PUBMED.pth'%l, map_location='cuda')
        PRO_VECS.append(eig_vecs0)

    writer = SummaryWriter('tensorboard_0319/0322-MoMoE_full_dimension-3072ffns-longtaile-pubmed-2e-4-checkpoint40000-medical-longtailed-justhead%d-1e-4-%dEPOCH-lessdata-TESTTT'%(k,EPOCH))
    # writer = SummaryWriter('tensorboard_0223/0301_MoMoE_MIXED_LEGAL_PUBMED_correct2_GDS_finetuned-64*128-GAD_single1-2.5e-5-PR')
    
    
    # writer = SummaryWriter('tensorboard_0223/0225_BERT_MIXED_LEGAL_PUBMED_finetuned-64*128-Overruling')

    optimizer = optim.AdamW([
        {'params': model.parameters()},
        # {'params': model.head.parameters()},
        # {'params': model.pooler.parameters()}
    ], lr=1e-4, weight_decay=0.02, betas=[0.9, 0.99], eps=1e-6)
    model, optimizer, train_loader, test_loader,val_loader_longtailed = accelerator.prepare(model, optimizer, train_loader, test_loader,val_loader_longtailed)
    num_epoch = EPOCH
    device2 = torch.device("cuda:2")
    
    model0.to(device2)
    model.to(device)
    model0.eval()

    for param in model.embeddings.parameters():
        param.requires_grad = False
    for param in model.layers.parameters():
        param.requires_grad = False
    # for param in model.bert_layers.parameters():
    #     param.requires_grad = False
    for epoch in range(num_epoch):
        
        losses = []
        for i, batch in enumerate(train_loader):
            # loss, _ = model(**batch)
            # print(i)
            # if i == 9:
            #     break
            batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            # print(epoch, i)
            # # print(next(model.parameters()).device)
            # # for key, tensor in batch.items():
            # #     print(f"{key} is on {tensor.device}")
            # # _, _, layer_outputs,_ = router(**batch)
            
            
            _,_,_,_,_,_,inputs = model0(batch0['input_ids'],batch0['attention_mask'], batch0['label'])
            # # hidden_states_for_router.append(router.bert.embeddings(batch['input_ids']).to(device))
            # # hidden_states_for_router = hidden_states_for_router  + layer_outputs[0:-1]
            # # print(len(hidden_states_for_router))
            inputs = [i0.to(device) for i0 in inputs]
            
            loss,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['label'], cluster_centers,inputs,PRO_VECS)
            # loss,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['label'], cluster_centers2,inputs,PRO_VECS)


            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])

        losses = []
        preds = []
        P = []
        R = []
        with torch.no_grad():
            for j, batch in enumerate(test_loader):

                batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
                batch = {key: tensor.to(device) for key, tensor in batch.items()}
                
                _,_,_,_,_,_,inputs = model0(batch0['input_ids'],batch0['attention_mask'], batch0['label'])
                

                inputs = [i0.to(device) for i0 in inputs]
                
                loss,score,ids = model(batch['input_ids'],batch['attention_mask'], batch['label'], cluster_centers,inputs,PRO_VECS)

                y_pred = torch.argmax(score, dim=-1)
                pred = y_pred == batch['label']
                true_positives = torch.sum(y_pred * batch['label'])
                predicted_positives = torch.sum(y_pred)
                actual_positives = torch.sum(batch['label'])
                precision = (true_positives+1e-7) / (predicted_positives+1e-7)
                recall = (true_positives+1e-7) / (actual_positives+1e-7)
                P.append(accelerator.gather(precision))
                R.append(accelerator.gather(recall))
                
                losses.append(accelerator.gather(loss.repeat(config.batch_size)))
                preds.append(accelerator.gather(pred))
        loss_test = torch.mean(torch.cat(losses)[:len(test_loader.dataset)])
        preds = torch.cat(preds)
        acc_test = torch.sum(preds) / len(preds)
        # print(R)
        P = torch.tensor(P)
        P_S = torch.sum(P) / len(P)

        R = torch.tensor(R)
        R_S = torch.sum(R) / len(R)

        F1 = 2*P_S*R_S/(P_S+R_S)



        losses0 = []
        preds0 = []
        P0 = []
        R0 = []
        with torch.no_grad():
            for j, batch in enumerate(val_loader_longtailed):

                batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
                batch = {key: tensor.to(device) for key, tensor in batch.items()}
                
                _,_,_,_,_,_,inputs = model0(batch0['input_ids'],batch0['attention_mask'], batch0['label'])
                

                inputs = [i0.to(device) for i0 in inputs]
                
                loss,score,ids = model(batch['input_ids'],batch['attention_mask'], batch['label'], cluster_centers,inputs,PRO_VECS)

                y_pred = torch.argmax(score, dim=-1)
                pred = y_pred == batch['label']
                true_positives = torch.sum(y_pred * batch['label'])
                predicted_positives = torch.sum(y_pred)
                actual_positives = torch.sum(batch['label'])
                precision = (true_positives+1e-7) / (predicted_positives+1e-7)
                recall = (true_positives+1e-7) / (actual_positives+1e-7)
                P0.append(accelerator.gather(precision))
                R0.append(accelerator.gather(recall))
                
                losses0.append(accelerator.gather(loss.repeat(config.batch_size)))
                preds0.append(accelerator.gather(pred))
        loss_test0 = torch.mean(torch.cat(losses0)[:len(val_loader_longtailed.dataset)])
        preds0 = torch.cat(preds0)
        acc_test0 = torch.sum(preds0) / len(preds0)
        # print(R)
        P0 = torch.tensor(P0)
        P_S0 = torch.sum(P0) / len(P0)

        R0 = torch.tensor(R0)
        R_S0 = torch.sum(R0) / len(R0)

        F10 = 2*P_S0*R_S0/(P_S0+R_S0)
        accelerator.print(f'Epoch:{epoch} ({(i + 1) * accelerator.num_processes} Updates), Train Loss: {loss_train}, Test Loss: {loss_test}, Test Acc: {acc_test}, IDS: {ids}')

        if accelerator.is_local_main_process:
            writer.add_scalar('acc_test', acc_test, epoch)
            writer.add_scalar('loss_train', loss_train, epoch)
            writer.add_scalar('P', P_S, epoch)
            writer.add_scalar('R', R_S, epoch)
            writer.add_scalar('F1', F1, epoch)
            writer.add_scalar('longtailed-acc_test', acc_test0, epoch)
            writer.add_scalar('longtailed-P', P_S0, epoch)
            writer.add_scalar('longtailed-R', R_S0, epoch)
            writer.add_scalar('longtailed-F1', F10, epoch)



if __name__ == "__main__":
    config = BertConfig.from_json_file('config/MoMoE.json')
    config0 = BertConfig.from_json_file('config/bert.json')
    EPOCH = 30
    # PO= np.zeros(EPOCH)
    # RO= np.zeros(EPOCH)

    # for k in range(1,11):
    #     # dataset = GAD(config)
    #     dataset = GAD_single(config,k)




    #     torch.cuda.set_device(1)
    #     device = torch.device("cuda")
    #     model = MoMoEForCLS_every(config)
    #     # model = BERTForCLS(config)

    #     finetune(model, dataset,k,EPOCH)
    dataset = medical_abstract(config)
    # dataset = Overruling(config)


    k = 1
    torch.cuda.set_device(1)
    device = torch.device("cuda")
    
    model = MoMoEForCLS(config)
    # model = BERTForCLS(config)

    finetune(model, dataset,k,EPOCH)
    # print(PO/10.0,RO/10.0)