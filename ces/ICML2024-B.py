import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import RestaurantforLM_1103, MixedData, MixedData_stage1, ACLForLM,old_MixedData_after_stage1, Mixdata_1103, Wikitext,ACLForLM_1103,Mixdata_1115,Review_1103,Wikitxt103ForLM_1103,MixedData_1211,Wikitxt103ForLM_0102_bert
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
            loss,_, _,_,_,_ = model(batch['input_ids'],batch['attention_mask'],batch['labels'],routes)
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

    angel = 0
    ordupdata = 20
    equal_in_params = 0


    lambdaW = config.lambdaW
    lambdaF = config.lambdaF

    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    val_loader2 = ahead1.val_loader
    # val_loader3 = ahead2.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    writer = SummaryWriter('0116-angel-justinhibition-allres-MixedData_1211')
    router_loss_fn = nn.CrossEntropyLoss().to(device)
    num_updates = num_epochs * len(train_loader)

    if equal_in_params:
        for l in range(12):
            model.layers.layers[l].experts[1].load_state_dict(model.layers.layers[l].experts[0].state_dict())


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
        route = [1 for i in range(int(config.batch_size/2))]+[0 for i in range(int(config.batch_size/2))]
        # print(route)
        random.shuffle(route)
        # print(route)
        routes = [route for i in range(config.num_hidden_layers)]
        # print(routes)



        for i, batch in enumerate(train_loader):
            # print(batch['attention_mask'].shape)
            # if i>=120:
            #     break
            if i%60 == 0:

                print(i,model.layers.layers[0].state, model.layers.layers[0].k,len(model.layers.layers[0].out_for_router[0]))
            # if i == 3:
            #     break
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            # print(next(model.parameters()).device)
            # for key, tensor in batch.items():
            #     print(f"{key} is on {tensor.device}")
            loss,_, OUT_OF_ROUTER,ROUTES,router_labels,_ = model(batch['input_ids'],batch['attention_mask'],batch['labels'],routes)
            # print("OUT_FOR_ROUTER: ",len(OUT_FOR_ROUTER[0]))
            # if model.layers.layers[0].state:
            #     print(ROUTES)
            if model.layers.layers[0].state:
                for r,l in zip(OUT_OF_ROUTER,router_labels):

                    # print(l.shape,r.shape)
                    router_loss = router_loss_fn(r,l.to(device))
                    router_losses.append(accelerator.gather(router_loss.repeat(config.batch_size)))
                    router_loss.backward()
                        
                # print("routerlabel",router_labels)
                # print("routerloss",router_loss)
                        
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()    
        
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        # loss_Router = torch.mean(torch.cat(router_losses)[:len(router_losses)])
        loss_Router = 0
        loss_valid = validate(model, val_loader, accelerator, device,routes)
        # loss_valid2 = validate(model, val_loader2, accelerator, device)
        # accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Ahead Valid Loss: {loss_valid2}')
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, ROUTER LOSS:{loss_Router}')
        CENTERS = {}
        if epoch == num_epochs-1:

            for l in range(12):
                # print(model.layers.layers[l].centers)
                CENTERS[l] = model.layers.layers[l].centers
            torch.save(CENTERS,'0108-angel-justinhibition-combineres-centers-withoutupdatacenters.pth')

        if accelerator.is_local_main_process:
            writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
            # writer.add_scalar('perplexity_valid', loss_valid, epoch)
            # writer.add_scalar('ahead_perplexity_valid', loss_valid2, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    # accelerator.save_state('./bert-1103-stage0')
    
    
    # torch.save(model,'0108-angel-justinhibition-combineres-equal-inparam.pth')
    torch.save(model,'0116-angel-justinhibition-allres-MixedData_1211.pth')

    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/AngeL_MoE.json')
    # dataset = RestaurantForLM_small(config=config)
    dataset =MixedData_1211(config = config)
    ahead1 = MixedData_1211(config)
    # ahead2 = Mixdata_1103(config)
    # ahead_dataset = old_MixedData_after_stage1(config = config)
    
    torch.cuda.set_device(3)
    device = torch.device("cuda")
    model = base_models.AngeLB_MoE(config=config)
    model.to(device)
    # model = nn.DataParallel(model)
    
    train(model=model, num_epochs=50, dataset=dataset, device=device,ahead1=ahead1)