import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForMaskedLM
from transformer.Transformer import MoMoE_layer_0731_narrowneck,MoMoE_layer_0430_1_narrowneck,MoMoE_layer_0430_2_narrowneck,MoMoE_layer_0514_narrowneck,MoMoE_layer_0514_2_narrowneck,MoMoE_layer_0430_narrowneck,MoMoE_layer_0418_narrowneck,MoMoE_layer_0419_narrowneck,MoMoE_layer_0413_narrowneck,MoMoE_layer_0411_narrowneck,TransformerEncoder,MoMoE_layer_0306_narrowneck_adl,MoMoE_layer_0306_narrowneck,BertModel,Embeddings,MoMoE_layer_uniqueatt_commonffn,MoMoE_layer_0229,MoMoE_layer_0226
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertOnlyNSPHead
from transformers import BertConfig, get_cosine_schedule_with_warmup
from accelerate import Accelerator,load_checkpoint_and_dispatch
from accelerate import DistributedDataParallelKwargs as DDPK
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from Dataset import GLUE,SST2_w
from Dataset_new import GAD,Overruling,GAD_single,medical_abstract,multi_domains,Casehold,EUADR,SST2,COLA,MRPC,QQP,RTE,QNLI
import util
import json
import base_models
import numpy as np
import random
import pandas as pd
import os


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    return layer_data


class MoMoEForCLS_0412(nn.Module):
    def __init__(self, config) -> None:
        super(MoMoEForCLS_0412, self).__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 2 or i == config.num_hidden_layers - 1:
                layers += [MoMoE_layer_0411_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        # self.bert_layers = nn.ModuleList([TransformerEncoder(config) for _ in range(10)])
        # self.layers = nn.ModuleList([MoMoE_layer_0306_narrowneck_adl(config) for i in range(2)])
        self.embeddings = Embeddings(config)
        self.pooler = BertPooler(config)
        self.head = nn.Linear(config.hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()
        # self.deval = 1
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
    
    def forward(self, input_ids, attention_mask, label, W_IDS):
        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0411_narrowneck):
                hidden_states = layer(hidden_states, attention_mask, W_IDS)
            else:
                raise ModuleNotFoundError
        pooled_output = self.pooler(hidden_states)
        score = self.head(pooled_output)
        cls_loss = self.criterion(score, label)
        # print(expert_ids)
        return cls_loss, score


class MoMoEForCLS_0731(nn.Module):
    def __init__(self, config) -> None:
        super(MoMoEForCLS_0731, self).__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 2 or i == config.num_hidden_layers - 1:
                layers += [MoMoE_layer_0731_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        # self.bert_layers = nn.ModuleList([TransformerEncoder(config) for _ in range(10)])
        # self.layers = nn.ModuleList([MoMoE_layer_0306_narrowneck_adl(config) for i in range(2)])
        self.embeddings = Embeddings(config)
        self.pooler = BertPooler(config)
        self.head = nn.Linear(config.hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()
        # self.deval = 1
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
    
    def forward(self, input_ids, attention_mask, label, W_IDS):
        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states,_ = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0731_narrowneck):
                hidden_states = layer(hidden_states, attention_mask, W_IDS)
            else:
                raise ModuleNotFoundError
        pooled_output = self.pooler(hidden_states)
        score = self.head(pooled_output)
        cls_loss = self.criterion(score, label)
        # print(expert_ids)
        return cls_loss, score
class MoMoEForCLS_0514(nn.Module):
    def __init__(self, config) -> None:
        super(MoMoEForCLS_0514, self).__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 2 or i == config.num_hidden_layers - 1:
                layers += [MoMoE_layer_0514_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        # self.bert_layers = nn.ModuleList([TransformerEncoder(config) for _ in range(10)])
        # self.layers = nn.ModuleList([MoMoE_layer_0306_narrowneck_adl(config) for i in range(2)])
        self.embeddings = Embeddings(config)
        self.pooler = BertPooler(config)
        self.head = nn.Linear(config.hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()
        # self.deval = 1
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
    
    def forward(self, input_ids, attention_mask, label,ws):
        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states,_ = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0514_narrowneck):
                hidden_states = layer(hidden_states, attention_mask,ws)
            else:
                raise ModuleNotFoundError
        pooled_output = self.pooler(hidden_states)
        score = self.head(pooled_output)
        cls_loss = self.criterion(score, label)
        # print(expert_ids)
        return cls_loss, score



class MoMoEForCLS_0514_2(nn.Module):
    def __init__(self, config) -> None:
        super(MoMoEForCLS_0514_2, self).__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 1 or i == config.num_hidden_layers - 2:
                layers += [MoMoE_layer_0514_2_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        # self.bert_layers = nn.ModuleList([TransformerEncoder(config) for _ in range(10)])
        # self.layers = nn.ModuleList([MoMoE_layer_0306_narrowneck_adl(config) for i in range(2)])
        self.embeddings = Embeddings(config)
        self.pooler = BertPooler(config)
        self.head = nn.Linear(config.hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()
        # self.deval = 1
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
    
    def forward(self, input_ids, attention_mask, label,ws):
        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states,_ = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0514_2_narrowneck):
                hidden_states = layer(hidden_states, attention_mask,ws)
            else:
                raise ModuleNotFoundError
        pooled_output = self.pooler(hidden_states)
        score = self.head(pooled_output)
        cls_loss = self.criterion(score, label)
        # print(expert_ids)
        return cls_loss, score




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
        # outputs = []
        # att_outs = []
        # ffn_outs = []
        # att_self = []
        expert_ids = []
        # inputs.append(hidden_states)
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i],hi_ou[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,cat_att = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hi_ou[i])
            
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
        self.layers = nn.ModuleList([MoMoE_layer_0229(config) for i in range(config.num_hidden_layers)])
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
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,cat_att = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hi_ou[i],pro_vecs[i])
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



def finetune(model: MoMoEForCLS, dataset:SST2_w,k,EPOCH,task,CKPTS,lr,seed,LEGAL):
    device2 = torch.device("cuda:4")
    accelerator = Accelerator()
    train_loader = dataset.train_loader
    test_loader = dataset.val_loader
    # val_loader1, val_loader2,val_loader3,val_loader4= dataset.val_loader1,dataset.val_loader2,dataset.val_loader3,dataset.val_loader4
    model0 = base_models.MoMoE_0514_2(config=config)
    # model0 = base_models.MoMoE_0731(config=config)

    # load_checkpoint_and_dispatch(model0, '/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0521/MOA',device_map={"":device})

    load_checkpoint_and_dispatch(model0, '/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_REBUTALL/mof-copyparams',device_map={"":device})
    # print(model0)/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_REBUTALL/MOAOE3-%d
    # model01 = torch.load('0226-bert-batch64_seq128-MIXED_LEGAL_PUBMED-pretrain-lr2e-4-for-routing2.pth')
    model.layers.load_state_dict(model0.layers.state_dict())
    model.embeddings.load_state_dict(model0.embeddings.state_dict())
    cluster_centers = load_layer_data('0518-layer_centers-ffn-4t-longtail.pth')
    # cluster_centers = load_layer_data('0517-layer_centers-3t-longtail.pth')

    lora_mat = torch.load('0514-lora_mat.pth')
    cluster_radius=load_layer_data('0516-layer_radius-3t-longtail.pth')


    writer = SummaryWriter('/home/jxzhou/PLM_PER/BERT-CL-main/rebuttal_finetune/mof-copyparams-task%s'%task)
    center_model = base_models.BertForMLM(config=config)
    checkpoint = torch.load(os.path.join('MODELS_0514/BERT-768ffns-mixed-small-1.5e-4', 'pytorch_model.bin'))
    center_model.load_state_dict(checkpoint)
    

    optimizer = optim.AdamW([
        {'params': model.parameters()},

    ], lr=lr, weight_decay=0.02, betas=[0.9, 0.99], eps=1e-6)
    for param in model.layers.parameters():
        param.requires_grad = False
    for param in model.embeddings.parameters():
        param.requires_grad = False
    # model, optimizer, train_loader, test_loader,val_loader1, val_loader2,val_loader3,val_loader4 = accelerator.prepare(model, optimizer, train_loader, test_loader,val_loader1, val_loader2,val_loader3,val_loader4)
    # val_loader = [val_loader1, val_loader2,val_loader3,val_loader4]
    model, optimizer, train_loader, test_loader , center_model,cluster_centers[-1],lora_mat,cluster_radius[-1]= accelerator.prepare(model, optimizer, train_loader, test_loader,center_model,cluster_centers[-1],lora_mat,cluster_radius[-1])
    center_model.eval()
    num_epoch = EPOCH


    
    # model0.to(device2)
    # model.to(device)
    # model0.eval()
    for epoch in range(num_epoch):
        
        losses = []
        preds = []

        for i, batch in enumerate(train_loader):
            # if i==9:break
            # print(i)
            with torch.no_grad():   
                

                _,hidd = center_model.bert(batch['input_ids'],batch['attention_mask'])
                hidd = hidd.mean(1)
                # hidd = center_model.bert(batch['input_ids'],batch['attention_mask']).mean(1)
                hidd = torch.matmul(hidd,lora_mat.to(hidd.device))
                # print(cluster_centers[-1].shape)

                distances = torch.cdist(hidd.detach(), cluster_centers[-1])
                ws = torch.argmin(distances, dim=1)


            loss,_ = model(batch['input_ids'],batch['attention_mask'], batch['label'],ws)
            # loss,_ = model(batch['input_ids'],batch['attention_mask'], batch['label'])



            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        losses = []
        with torch.no_grad():

            for J, batch in enumerate(test_loader):

                # print(batch['input_ids'])
                with torch.no_grad():   
                

                    _,hidd = center_model.bert(batch['input_ids'],batch['attention_mask'])
                    hidd = hidd.mean(1)
                    # hidd = center_model.bert(batch['input_ids'],batch['attention_mask']).mean(1)
                    hidd = torch.matmul(hidd,lora_mat.to(hidd.device))
                    # print(cluster_centers[-1].shape)

                    distances = torch.cdist(hidd.detach(), cluster_centers[-1])
                    ws = torch.argmin(distances, dim=1)

                loss,score = model(batch['input_ids'],batch['attention_mask'], batch['label'],ws)
                # loss,score = model(batch['input_ids'],batch['attention_mask'], batch['label'])

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
        #             loss,score = model(batch['input_ids'][b].view(1,-1),batch['attention_mask'][b].view(1,-1), batch['label'][b].view(-1),batch['w_ids'][b].view(-1))
        #             y_pred = torch.argmax(score, dim=-1)
        #             pred = y_pred == batch['label'][b]

        #             # losses.append(accelerator.gather(loss.repeat(config.batch_size)))
        #             # preds.append(accelerator.gather(pred))
        #             long_acc.append(pred)
            # loss_test = torch.mean(torch.cat(losses)[:len(test_loader.dataset)])
            # preds = torch.cat(preds)
            # acc_test = torch.sum(preds) / len(preds)
        # print(lens)
        # print(long_acc)
        # df = pd.DataFrame({
        #     'Column1': [x.item() for x in lens],
        #     'Column2': [x.item() for x in long_acc]
        # })

        # # 保存DataFrame到CSV文件
        # df.to_csv('0428-%s-%d.csv'%(task,epoch), index=True)
        # LOSSES = []
        # ACCS = []

        # with torch.no_grad():
        #     for d in range(4):
        #         losses = []
        #         preds = []
        #         for J, batch in enumerate(val_loader[d]):
        #             # if J == 9:
        #             #     break
        #             loss,score,_ = model(batch['input_ids'],batch['attention_mask'], batch['label'],batch['w_ids'])
        #             y_pred = torch.argmax(score, dim=-1)
        #             # print(y_pred)
        #             pred = y_pred == batch['label']
        #             # true_positives = torch.sum(y_pred * batch['label'])
        #             # # print(true_positives)
        #             # predicted_positives = torch.sum(y_pred)
        #             # actual_positives = torch.sum(batch['label'])
        #             # print(actual_positives)
        #             # precision = (true_positives+1e-7) / (predicted_positives+1e-7)
        #             # recall = (true_positives+1e-7) / (actual_positives+1e-7)
        #             # P.append(accelerator.gather(precision))
        #             # R.append(accelerator.gather(recall))
        #             losses.append(accelerator.gather(loss.repeat(config.batch_size)))
        #             preds.append(accelerator.gather(pred))
        #         loss_test = torch.mean(torch.cat(losses)[:len(test_loader.dataset)])
        #         preds = torch.cat(preds)
        #         acc_test = torch.sum(preds) / len(preds)

        #         LOSSES.append(loss_test)
        #         ACCS.append(acc_test)


        # loss_test = sum(LOSSES)/len(LOSSES)
        # acc_test = sum(ACCS)/len(ACCS)

        accelerator.print(f'Epoch:{epoch} ({(i + 1) * accelerator.num_processes} Updates), Train Loss: {loss_train}, Test Loss: {loss_test}, Test Acc: {acc_test}')

        
        if accelerator.is_local_main_process:
            writer.add_scalar('acc_test', acc_test, epoch)
            writer.add_scalar('loss_train', loss_train, epoch)
            writer.add_scalar('loss_val', loss_test, epoch)

            # for i in range(4):
            #     writer.add_scalar('accuracy-dataset%d'%i, ACCS[i], epoch)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    seed = 14
    set_seed(seed)
    config = BertConfig.from_json_file('config/MoMoE.json')
    config0 = BertConfig.from_json_file('config/bert.json')
    EPOCH = 10
    datasets = {}
    datasets['casehold'] = Casehold(config)
    datasets['overruling'] = Overruling(config)

    datasets['sst2'] = GLUE(config)

    datasets['gad'] = GAD(config)
    datasets['euadr'] = EUADR(config)
    # datasets['cola'] = COLA(config)
    # datasets['mrpc'] = MRPC(config)

    # datasets['gnli'] = QNLI(config)

    # datasets['rte'] = RTE(config)
    # datasets['qqp'] = QQP(config)

    k = 1
    torch.cuda.set_device(2)

    device = torch.device("cuda:2")
    LEGAL = 80000
    # device = torch.device("cuda:2")
    model = MoMoEForCLS_0514_2(config)
    # model = MoMoEForCLS_0731(config)


    for i in datasets:
        print(i)

        finetune(model, datasets[i],k,EPOCH,i,80000,1e-3,seed,LEGAL)



