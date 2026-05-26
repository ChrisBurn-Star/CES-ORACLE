import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForMaskedLM
from transformer.Transformer_MOE import BertModel,BertModel_every2layers
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertOnlyNSPHead
from transformers import BertConfig, get_cosine_schedule_with_warmup
from accelerate import Accelerator,load_checkpoint_and_dispatch
from accelerate import DistributedDataParallelKwargs as DDPK
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# from Dataset import GLUE
import util
import json
import pandas as pd
from Dataset import GLUE
from Dataset_new import GAD,Overruling,GAD_single,medical_abstract,multi_domains,Casehold,EUADR,SST2,COLA,MRPC,QQP,RTE,QNLI
import numpy as np
import random
def get_long_data(tensor):
    not_ones = tensor != 1

    # 计算不为1的元素个数
    count_not_ones = torch.sum(not_ones)

    

    return count_not_ones
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
        output,att_sself,O,_,_,_ = self.bert(input_ids, attention_mask)
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
        output,_,_,_,_,_ = self.bert(input_ids, attention_mask)
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
        output,_,_,_,_,_ = self.bert(input_ids, attention_mask)
        pooled_output = self.pooler(output)
        score = self.head(pooled_output)
        cls_loss = self.criterion(score, label)

        return cls_loss, score


def finetune(model: MoEForCLS, dataset: GLUE,k,EPOCH,task,CKPTS,lr,seed,LEGAL):
    accelerator = Accelerator()
    train_loader = dataset.train_loader
    test_loader = dataset.val_loader
    # val_loader1, val_loader2,val_loader3,val_loader4= dataset.val_loader1,dataset.val_loader2,dataset.val_loader3,dataset.val_loader4
    # val_loader_longtailed = dataset.val_loader_longtailed
    # accelerator.load_state()
    model0 = BertForMLM_every2layers(config)
    # model0 = torch.load("/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0315/0320-MoE-3072ffns-longtailed-pubmed-150+350W*1-2e-4-checkpoints40000.pth")
    # model0 = torch.load("0315-MoE-768ffns-150+350W*1-3e-4.pth")
    # model.bert.load_state_dict(model0.bert.state_dict())
    
    load_checkpoint_and_dispatch(model0, '/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_REBUTALL/moe-copyparams',device_map={"":device})

    model.bert.load_state_dict(model0.bert.state_dict())
    # print(111)
    




    writer = SummaryWriter('/home/jxzhou/PLM_PER/BERT-CL-main/rebuttal_finetune/moe-copyparams-task%s'%task)

    optimizer = optim.AdamW([
        {'params': model.parameters()},
        # {'params': model.head.parameters()},
        # {'params': model.pooler.parameters()}
    ], lr=lr, weight_decay=0.02, betas=[0.9, 0.99], eps=1e-6)
    for param in model.bert.parameters():
        param.requires_grad = False
    model, optimizer, train_loader, test_loader= accelerator.prepare(model, optimizer, train_loader, test_loader)
    
    # model, optimizer, train_loader, test_loader,val_loader1,val_loader2,val_loader3,val_loader4= accelerator.prepare(model, optimizer, train_loader, test_loader,val_loader1,val_loader2,val_loader3,val_loader4)
    num_epoch = EPOCH
    # val_loader = [val_loader1,val_loader2,val_loader3,val_loader4]


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

            # writer.add_scalar('P', P_S, epoch)
            # writer.add_scalar('R', R_S, epoch)
            # writer.add_scalar('F1', F1, epoch)
            # writer.add_scalar('longtailed-acc_test', acc_test0, epoch)
            # writer.add_scalar('longtailed-P', P_S0, epoch)
            # writer.add_scalar('longtailed-R', R_S0, epoch)
            # writer.add_scalar('longtailed-F1', F10, epoch)
# 
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    config = BertConfig.from_json_file('config/MoMoE.json')
    # device = torch.device("cuda")
    # torch.cuda.set_device(2)
    seed =14
    LEGAL = 90000
    set_seed(seed)

    torch.cuda.set_device(7)

    device = torch.device("cuda:7")


    # dataset = GAD(config)
    # torch.cuda.set_device(6)
    EPOCH = 10
    # PO= np.zeros(EPOCH)
    # RO= np.zeros(EPOCH)
    # for k in range(1,11):
    #     dataset = GAD_single(config,k)

    #     # dataset = GLUE(config)
    #     # dataset = Overruling(config)



    #     model = MoEForCLS(config)

    #     finetune(model, dataset,k,EPOCH)
    # dataset = medical_abstract(config)

    datasets = {}
    # datasets['casehold'] = Casehold(config)
    # datasets['overruling'] = Overruling(config)

    # datasets['sst2'] = GLUE(config)

    datasets['gad'] = GAD(config)
    datasets['euadr'] = EUADR(config)

    # datasets['cola'] = COLA(config)
    # datasets['mrpc'] = MRPC(config)

    # datasets['gnli'] = QNLI(config)

    # datasets['rte'] = RTE(config)
    # datasets['qqp'] = QQP(config)

    k = 1


    

    model = MoEForCLS_every(config)
    for i in datasets:
        print(i)

        finetune(model, datasets[i],k,EPOCH,i,80000,1e-3,seed,LEGAL)
        # print(PO/10.0,RO/10.0)