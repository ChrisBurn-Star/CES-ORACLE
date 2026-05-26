import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForMaskedLM
from transformer.Transformer_MOE import BertModel,BertModel_every2layers
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertOnlyNSPHead
from transformers import BertConfig, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from Dataset import GLUE
import util
import json

from Dataset_new import GAD,Overruling,GAD_single,legal_argument_mining,medical_abstract
import numpy as np
class BertForMLM_every2layers(nn.Module):
    def __init__(self, config):
        super(BertForMLM_every2layers, self).__init__()
        self.config = config
        self.bert = BertModel_every2layers(config)
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
        output,att_sself,O,_ = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores,att_sself,O


class BertForMLM(nn.Module):
    def __init__(self, config):
        super(BertForMLM, self).__init__()
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
        output,att_sself,O,_ = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores,att_sself,O


class MoEForCLS(nn.Module):
    def __init__(self, config) -> None:
        super(MoEForCLS, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.pooler = BertPooler(config)
        self.head = nn.Linear(config.hidden_size, 5)
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
        output,_,_,_ = self.bert(input_ids, attention_mask)
        pooled_output = self.pooler(output)
        score = self.head(pooled_output)
        cls_loss = self.criterion(score, label)

        return cls_loss, score



class MoEForCLS_every(nn.Module):
    def __init__(self, config) -> None:
        super(MoEForCLS_every, self).__init__()
        self.config = config
        self.bert = BertModel_every2layers(config)
        self.pooler = BertPooler(config)
        self.head = nn.Linear(config.hidden_size, 5)
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
        output,_,_,_ = self.bert(input_ids, attention_mask)
        pooled_output = self.pooler(output)
        score = self.head(pooled_output)
        cls_loss = self.criterion(score, label)

        return cls_loss, score


def finetune(model: MoEForCLS, dataset: GLUE,k,EPOCH):
    accelerator = Accelerator()
    train_loader = dataset.train_loader
    test_loader = dataset.val_loader
    val_loader_longtailed = dataset.val_loader_longtailed

    model0 = torch.load("/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0315/0320-MoE-3072ffns-longtailed-pubmed-150+350W*1-2e-4-checkpoints40000.pth")
    # model0 = torch.load("0315-MoE-768ffns-150+350W*1-3e-4.pth")
    model.bert.load_state_dict(model0.bert.state_dict())



    writer = SummaryWriter('tensorboard_0319/0322-MoE-3072ffns-longtailed-pubmed-150+350W*1-2e-4-checkpoints40000-medical-longtailed-justhead%d-1e-4-%dEPOCHS-lessdata'%(k,EPOCH))

    optimizer = optim.AdamW([
        {'params': model.parameters()},
        # {'params': model.head.parameters()},
        # {'params': model.pooler.parameters()}
    ], lr=1e-4, weight_decay=0.02, betas=[0.9, 0.99], eps=1e-6)
    model, optimizer, train_loader, test_loader,val_loader_longtailed = accelerator.prepare(model, optimizer, train_loader, test_loader, val_loader_longtailed)
    num_epoch = EPOCH

    for param in model.bert.parameters():
        param.requires_grad = False
    for epoch in range(num_epoch):
        
        losses = []
        losses0 = []

        for i, batch in enumerate(train_loader):
            # if i == 9:
            #     break
            # print(batch)

            loss, _ = model(**batch)
            # print(i)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])

        losses = []
        preds = []
        P = []
        R = []
        losses0 = []
        preds0 = []
        P0 = []
        R0 = []
        with torch.no_grad():
            for J, batch in enumerate(test_loader):
                # if J == 9:
                #     break
                loss, score = model(**batch)
                y_pred = torch.argmax(score, dim=-1)
                # print(y_pred)
                pred = y_pred == batch['label']
                true_positives = torch.sum(y_pred * batch['label'])
                # print(true_positives)
                predicted_positives = torch.sum(y_pred)
                actual_positives = torch.sum(batch['label'])
                # print(actual_positives)
                precision = (true_positives+1e-7) / (predicted_positives+1e-7)
                recall = (true_positives+1e-7) / (actual_positives+1e-7)
                P.append(accelerator.gather(precision))
                R.append(accelerator.gather(recall))
                losses.append(accelerator.gather(loss.repeat(config.batch_size)))
                preds.append(accelerator.gather(pred))

        # with torch.no_grad():
            for J, batch in enumerate(val_loader_longtailed):
                # if J == 9:
                #     break
                loss, score = model(**batch)
                y_pred = torch.argmax(score, dim=-1)
                # print(y_pred)
                pred = y_pred == batch['label']
                true_positives = torch.sum(y_pred * batch['label'])
                # print(true_positives)
                predicted_positives = torch.sum(y_pred)
                actual_positives = torch.sum(batch['label'])
                # print(actual_positives)
                precision = (true_positives+1e-7) / (predicted_positives+1e-7)
                recall = (true_positives+1e-7) / (actual_positives+1e-7)
                P0.append(accelerator.gather(precision))
                R0.append(accelerator.gather(recall))
                losses0.append(accelerator.gather(loss.repeat(config.batch_size)))
                preds0.append(accelerator.gather(pred))
        loss_test0 = torch.mean(torch.cat(losses0)[:len(val_loader_longtailed.dataset)])
        preds0 = torch.cat(preds0)
        acc_test0 = torch.sum(preds0) / len(preds0)
        # print(P)
        P0 = torch.tensor(P0)
        P_S0 = torch.sum(P0) / len(P0)

        R0 = torch.tensor(R0)
        R_S0 = torch.sum(R0) / len(R0)

        F10 = 2*P_S0*R_S0/(P_S0+R_S0)
        # PO[epoch]+=P_S.item()
        # RO[epoch]+=R_S.item()
        loss_test = torch.mean(torch.cat(losses)[:len(test_loader.dataset)])
        preds = torch.cat(preds)
        acc_test = torch.sum(preds) / len(preds)
        # print(P)
        P = torch.tensor(P)
        P_S = torch.sum(P) / len(P)

        R = torch.tensor(R)
        R_S = torch.sum(R) / len(R)

        F1 = 2*P_S*R_S/(P_S+R_S)
        # PO[epoch]+=P_S.item()
        # RO[epoch]+=R_S.item()


        accelerator.print(f'Epoch:{epoch} ({(i + 1) * accelerator.num_processes} Updates), Train Loss: {loss_train}, Test Loss: {loss_test}, Test Acc: {acc_test}')
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
# 

if __name__ == "__main__":
    config = BertConfig.from_json_file('config/MoMoE.json')
    # dataset = GAD(config)
    torch.cuda.set_device(6)
    EPOCH = 30
    # PO= np.zeros(EPOCH)
    # RO= np.zeros(EPOCH)
    # for k in range(1,11):
    #     dataset = GAD_single(config,k)

    #     # dataset = GLUE(config)
    #     # dataset = Overruling(config)



    #     model = MoEForCLS(config)

    #     finetune(model, dataset,k,EPOCH)
    dataset = medical_abstract(config)
    # dataset = Overruling(config)
    k = 1


    

    model = MoEForCLS(config)

    finetune(model, dataset,k,EPOCH)
    # print(PO/10.0,RO/10.0)