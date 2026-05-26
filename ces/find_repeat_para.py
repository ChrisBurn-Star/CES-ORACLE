import torch
import torch.nn as nn
import torch.optim as optim
from transformer.Transformer_MOE import BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import BertConfig, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from Dataset import Wikitext
from Dataset_new import RestaurantforLM_1103,Review_1103,ACLForLM_1103,Mixdata_1115
from einops import rearrange
import util
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import base_models
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.profiling.flops_profiler import FlopsProfiler


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
        output = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores
def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    return layer_data
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb
def train(model: BertForMLM, dataset, ahead_dataset):
    # accelerator: https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
    # accelerator = Accelerator(kwargs_handlers=[DDPK(find_unused_parameters=True)])
    accelerator = Accelerator()
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
    # test_loader = dataset.test_loader
    ahead_val_loader = ahead_dataset.val_loader
    ahead_train_loader = ahead_dataset.train_loader
    config = model.config
    # model = torch.load('1127-bert-only-review.pth')
    # model = torch.load('1127-moe-form-yubin-review.pth')

    cluster_centers = load_layer_data('1115-layer_centers-2expert.pth')
    router = base_models.BertWithSavers(config=config)
    
    model0 = torch.load('1115-only-bert-formoe-stage0.pth')
    router.bert.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    for i in range(config.num_hidden_layers):
        router.bert.layers.layers[i].load_state_dict(model0.bert.encoders.layers[i].state_dict())
    router.head.load_state_dict(model0.head.state_dict())
    model = torch.load('1127_vermilion_model_stage2.pth')
    # model = torch.load('1120_rose_model_satge2-replay.pth')
    # model = torch.load('1115-only-bert-formoe-stage2.pth')


    lrs = 1e-5
    num_epochs = 1
    num_updates = num_epochs * len(train_loader)

    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_updates * 0.06,
        num_training_steps=num_updates,
    )
    
    model, optimizer, lr_scheduler, train_loader, val_loader, ahead_val_loader, ahead_train_loader, router = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, ahead_val_loader, ahead_train_loader,router)
    module = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    ACTIVED_PARAS1 = []
    ACTIVED_PARAS2 = []
    ACTIVED_TENSORS1 = []
    ACTIVED_TENSORS2 = []

    for epoch in range(num_epochs):
        c=0
        model.train()
        
        for i, batch in enumerate(ahead_train_loader):
            if i%125 ==0:
                print(i)
                c+=1
                # hidden_states_for_router = []

                # _, _, layer_outputs,_ = router(**batch)
                #     #
                
                # hidden_states_for_router.append(router.bert.embeddings(batch['input_ids']))
                # hidden_states_for_router = hidden_states_for_router  + layer_outputs[0:-1]
                # # print(epoch, i)
                # loss, _, _ ,_, _= model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers, hidden_states_for_router)

                # loss, _= model(**batch)
                loss, _, _ ,_, EXPERT_IDS= model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers)

                
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.all(param.grad == 0):
                            ACTIVED_PARAS1.append(0)
                            ACTIVED_TENSORS1.append(0)
                            pass
                        else:
                            ACTIVED_PARAS1.append(name)
                            # print(torch.sum(torch.eq(param.grad, 0)))
                            ACP = torch.where(torch.abs(param.grad) > lrs, torch.tensor(1), param.grad)
                            ACP = torch.where(torch.abs(ACP) <=lrs , torch.tensor(-1), ACP)
                            ACTIVED_TENSORS1.append(ACP)
                            # ACTIVED_PARAS1_tensor=torch.nonzero(param.grad)
                            # print(ACTIVED_PARAS1_tensor.shape[0])
                            # for k in range(ACTIVED_PARAS1_tensor.shape[0]):
                            #     ACTIVED_PARAS1.append(str(str(name)+str(ACTIVED_PARAS1_tensor[k])))
                            #     # print(str(str(name)+str(ACTIVED_PARAS1_tensor[k])))
                    else:
                        ACTIVED_PARAS1.append(0)
                        ACTIVED_TENSORS1.append(0)
                optimizer.zero_grad()

    for epoch in range(num_epochs):
        c=0
        model.train()

        for i, batch in enumerate(train_loader):
            if i %50 == 0:
                c+=1
                # hidden_states_for_router = []

                # _, _, layer_outputs,_ = router(**batch)
                #     #
                
                # hidden_states_for_router.append(router.bert.embeddings(batch['input_ids']))
                # hidden_states_for_router = hidden_states_for_router  + layer_outputs[0:-1]
                # # print(epoch, i)
                # loss, _, _ ,_, _= model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers, hidden_states_for_router)

                # loss, _ = model(**batch)
                loss, _, _ ,_, EXPERT_IDS= model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers)

                
                loss.backward()

                for name, param in model.named_parameters():

                    if param.grad is not None:
                        if torch.all(param.grad == 0):
                            ACTIVED_PARAS2.append(0)
                            ACTIVED_TENSORS2.append(0)
                        else:
                            ACTIVED_PARAS2.append(name)
                            
                            ACP = torch.where(torch.abs(param.grad) > lrs, torch.tensor(1), param.grad)
                            ACP = torch.where(torch.abs(ACP) <= lrs, torch.tensor(0), ACP)
                            ACTIVED_TENSORS2.append(ACP)
                    else:
                        ACTIVED_PARAS2.append(0)
                        ACTIVED_TENSORS2.append(0)
                            
                            # ACTIVED_PARAS2_tensor=torch.nonzero(param.grad)
                            # print(ACTIVED_PARAS1_tensor.shape[0])
                            # for k in range(ACTIVED_PARAS2_tensor.shape[0]):
                            #     ACTIVED_PARAS2.append(str(str(name)+str(ACTIVED_PARAS2_tensor[k])))
                optimizer.zero_grad()

    repeat_count = 0
    all_count = 0
    repeat_paras = []
    total_params = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad)

    for l in range(len(ACTIVED_PARAS2)):
        para_name = ACTIVED_PARAS2[l]
        if para_name == ACTIVED_PARAS1[l] and para_name != 0:
            all_count+=ACTIVED_TENSORS1[l].numel()
            TEMP = ACTIVED_TENSORS1[l] - ACTIVED_TENSORS2[l]
            # print(TEMP)
            repeat_count+= torch.sum(torch.eq(TEMP, 0))

            
    print(repeat_count)
    print(repeat_count/(total_params*c))
    # print(repeat_count/all_count)

                

def validate(model: BertForMLM, val_loader, accelerator):
    losses = []
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            loss, loss_dict = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity

if __name__ == "__main__":
    config = BertConfig.from_json_file('config/new_model.json')
    # model = base_models.rose_model(config)
    model = base_models.vermilion_model(config)

    # model = BertForMLM(config)


    dataset = ACLForLM_1103(config)
    # dataset = Review_1103(config)

    # ahead_dataset = RestaurantforLM_1103(config)
    ahead_dataset = Mixdata_1115(config)

    train(model, dataset, ahead_dataset)