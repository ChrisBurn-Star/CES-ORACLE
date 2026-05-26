import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import RestaurantforLM_1103, MixedData, MixedData_stage1, ACLForLM,old_MixedData_after_stage1, Mixdata_1103, Wikitext,ACLForLM_1103,Mixdata_1115,Review_1103,Wikitxt103ForLM_1103
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim

import torch
import numpy as np
import random


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


def validate(model, val_loader, accelerator, device,routes):
    losses = []
    for i, batch in enumerate(val_loader):  
        # print(i, model.layers.layers[0].state)
        batch = {key: tensor.to(device) for key, tensor in batch.items()}      
        with torch.no_grad():
            loss,_, _,_,_ = model(batch['input_ids'],batch['attention_mask'],batch['labels'],routes)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity


def get_gradient_norms(model):
    """Utility function to get gradient norms of a model."""
    return [param.grad.norm().item() for param in model.parameters() if param.grad is not None]


def train(model, num_epochs, dataset, device,ahead1):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    # model = torch.load('1127-bert-only.pth')

    angel = 3
    ordupdata = 20
    lambdaW = config.lambdaW
    lambdaF = config.lambdaF

    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    val_loader2 = ahead1.val_loader
    # val_loader3 = ahead2.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    writer = SummaryWriter('1225-angel3-test1-1e-3-prenorm')
    router_loss_fn = nn.CrossEntropyLoss().to(device)
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    # model, optimizer, lr_scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        
        """train origin bert (MLM only)"""
        losses = []
        wo_loss = []
        L1_loss = []
        router_losses = []
        all_loss =0.0
        router_loss = 0.0
        route = [1 for i in range(int(config.batch_size/2))]+[0 for i in range(int(config.batch_size/2))]
        routes = [route for i in range(config.num_hidden_layers)]



        for i, batch in enumerate(train_loader):
            # print(batch['attention_mask'].shape)
            if i%30 == 0:
                print(i, model.layers.layers[0].state, model.layers.layers[0].k)
            # if i == 30:
            #     break
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            # print(next(model.parameters()).device)
            # for key, tensor in batch.items():
            #     print(f"{key} is on {tensor.device}")
            loss,_, OUT_OF_ROUTER,ROUTES,router_labels = model(batch['input_ids'],batch['attention_mask'],batch['labels'],routes)
            # print("OUT_FOR_ROUTER: ",len(OUT_FOR_ROUTER[0]))
            if i%ordupdata == 0:
                if angel == 3:
                # W ORD LOSS
                    WI_loss = 0.0
                    w_c = 0
                    for l in range(config.num_hidden_layers):
                        for exp1 in range(config.num_experts-1):
                            for exp2 in range(exp1,config.num_experts):
                                
                                for head in range(config.num_attention_heads):
                                    Wi0 = model.layers.layers[l].experts[exp1].attention.self.heads[head].weight
                                    # print(Wi0[1].shape)
                                    Wi0 = Wi0.view(-1)

                                    Wj0 = model.layers.layers[l].experts[exp2].attention.self.heads[head].weight
                                    Wj0 = Wj0.view(-1)
                                    # print(Wi0.shape)


                                    WI_loss = WI_loss+(Wi0*Wj0).abs().sum()
                                    # print("Wi0*Wj0",(Wi0*Wj0).sum())
                                    w_c = w_c+1
                                    # print(WI_loss)
                                Wi1 = model.layers.layers[l].experts[exp1].attention.self.WO.weight.view(-1)

                                Wj1 = model.layers.layers[l].experts[exp2].attention.self.WO.weight.view(-1)
                                WI_loss = WI_loss+(Wi1*Wj1).abs().sum()
                                w_c = w_c+1




                    WI_loss = WI_loss/w_c
                    # FFN LI LOSS
                    FN_loss = 0.0
                    f_c = 0
                    for l in range(config.num_hidden_layers):
                        for exp in range(config.num_experts):
                            FNNP1 = sum(p.abs().sum() for p in model.layers.layers[l].experts[exp].ffn.parameters())
                            # FNNP2 = sum(p.abs().sum() for p in model.layers.layers[l].experts[exp].attention.parameters())
                            # FNNP = FNNP1+FNNP2
                            FNNP = FNNP1
                            FN_loss = FN_loss+ FNNP
                            f_c = f_c+1
                    FN_loss = FN_loss/f_c
                    
                    # print("wo loss",WI_loss)
                    # print("fo loss",FN_loss)
                    all_loss = loss+lambdaW*WI_loss+lambdaF*FN_loss
                    # print("all loss",all_loss)

                    # print(i, model.layers.layers[0].state)
                    wo_loss.append(accelerator.gather(WI_loss.repeat(config.batch_size)))
                    L1_loss.append(accelerator.gather(FN_loss.repeat(config.batch_size)))
                elif angel == 4:
                # W ORD LOSS
                    WI_loss = 0.0
                    w_c = 0
                    for l in range(config.num_hidden_layers):
                        for exp1 in range(config.num_experts-1):
                            for exp2 in range(exp1,config.num_experts):
                                
                                for head in range(config.num_attention_heads):
                                    Wi0 = model.layers.layers[l].experts[exp1].attention.self.heads[head].weight
                                    # print(Wi0[1].shape)
                                    Wi0 = Wi0.view(-1)

                                    Wj0 = model.layers.layers[l].experts[exp2].attention.self.heads[head].weight
                                    Wj0 = Wj0.view(-1)
                                    # print(Wi0.shape)


                                    WI_loss = WI_loss+(Wi0*Wj0).abs().sum()
                                    # print("Wi0*Wj0",(Wi0*Wj0).sum())
                                    w_c = w_c+1
                                    # print(WI_loss)
                                Wi1 = model.layers.layers[l].experts[exp1].attention.self.E.weight.view(-1)

                                Wj1 = model.layers.layers[l].experts[exp2].attention.self.E.weight.view(-1)
                                WI_loss = WI_loss+(Wi1*Wj1).abs().sum()
                                w_c = w_c+1
                                Wi2 = model.layers.layers[l].experts[exp1].attention.self.F.weight.view(-1)

                                Wj2 = model.layers.layers[l].experts[exp2].attention.self.F.weight.view(-1)
                                WI_loss = WI_loss+(Wi2*Wj2).abs().sum()

                                w_c = w_c+1




                    WI_loss = WI_loss/w_c
                    # FFN LI LOSS
                    FN_loss = 0.0
                    f_c = 0
                    for l in range(config.num_hidden_layers):
                        for exp in range(config.num_experts):
                            FNNP1 = sum(p.abs().sum() for p in model.layers.layers[l].experts[exp].ffn.parameters())
                            # FNNP2 = sum(p.abs().sum() for p in model.layers.layers[l].experts[exp].attention.parameters())
                            # FNNP = FNNP1+FNNP2
                            FNNP = FNNP1
                            FN_loss = FN_loss+ FNNP
                            f_c = f_c+1
                    FN_loss = FN_loss/f_c
                    
                    # print("wo loss",WI_loss)
                    # print("fo loss",FN_loss)
                    all_loss = loss+lambdaW*WI_loss+lambdaF*FN_loss
                    # print("all loss",all_loss)

                    # print(i, model.layers.layers[0].state)
                    wo_loss.append(accelerator.gather(WI_loss.repeat(config.batch_size)))
                    L1_loss.append(accelerator.gather(FN_loss.repeat(config.batch_size)))
                else:
                    pass
            else:
                all_loss = loss

            if model.layers.layers[0].state:
                c = 0
                for r,l in zip(OUT_OF_ROUTER,router_labels):
                    c+=1
                    # print(c)
                    # print(l.shape,r.shape)
                    router_loss = router_loss_fn(r,l.to(device))
                    router_losses.append(accelerator.gather(router_loss.repeat(config.batch_size)))
                    # optimizer.zero_grad()
                    router_loss.backward()
                        
                # print("routerlabel",router_labels)
                # print("routerloss",router_loss)
                        
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            
            optimizer.zero_grad()
            accelerator.backward(all_loss)
            optimizer.step()
            lr_scheduler.step()    
        
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_WO = torch.mean(torch.cat(wo_loss)[:len(wo_loss)])
        loss_L1 = torch.mean(torch.cat(L1_loss)[:len(L1_loss)])
        # loss_Router = torch.mean(torch.cat(router_losses)[:len(router_losses)])
        loss_Router = 0

        loss_valid = validate(model, val_loader, accelerator, device,routes)
        # loss_valid2 = validate(model, val_loader2, accelerator, device)
        # accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Ahead Valid Loss: {loss_valid2}')
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, W ORD LOSS:{loss_WO}, L1 LOSS:{loss_L1}, ROUTER LOSS:{loss_Router}')


        if accelerator.is_local_main_process:
            writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
            # writer.add_scalar('perplexity_valid', loss_valid, epoch)
            # writer.add_scalar('ahead_perplexity_valid', loss_valid2, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    # accelerator.save_state('./bert-1103-stage0')
    torch.save(model,'1226-angel3-test1-1e-3-prenorm.pth')
    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/AngeL_MoE.json')
    # dataset = RestaurantForLM_small(config=config)
    dataset = Wikitxt103ForLM_1103(config = config)
    ahead1 = Wikitxt103ForLM_1103(config)
    # ahead2 = Mixdata_1103(config)
    # ahead_dataset = old_MixedData_after_stage1(config = config)
    device = torch.device("cuda:1")
    # device = get_available_cuda_device()
    # print(device)
    model = base_models.AngeL3_MoE(config=config)
    model.to(device)
    # print(next(model.parameters()).device)
    # model = nn.DataParallel(model)
    
    train(model=model, num_epochs=50, dataset=dataset, device=device,ahead1=ahead1)