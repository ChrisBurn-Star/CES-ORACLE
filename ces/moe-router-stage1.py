import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import RestaurantForLM_small, MixedData,MixedData_stage1,Mixdata_1103,Mixdata_1115
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


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
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

def train(model, num_epochs, dataset, device, ahead_dataset, router ):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    
    freze_emb = 0 

    model0 = torch.load('1115-only-bert-formoe-stage0.pth')
    # cluster_centers = load_layer_data('layer_centers.pth')
    cluster_centers = load_layer_data('1115-layer_centers-2expert.pth')
    # print(len(cluster_centers), cluster_centers[9].shape)
    print()
    model.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    for i in range(config.num_hidden_layers):
        for j in range(config.num_experts):
            model.layers[i].experts[j].load_state_dict(model0.bert.encoders.layers[i].state_dict())
    model.head.load_state_dict(model0.head.state_dict())
    router.bert.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    for i in range(config.num_hidden_layers):
        router.bert.layers.layers[i].load_state_dict(model0.bert.encoders.layers[i].state_dict())
    router.head.load_state_dict(model0.head.state_dict())

    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    ahead_train_loader = ahead_dataset.train_loader
    ahead_val_loader = ahead_dataset.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    writer = SummaryWriter('1120_rose_model_satge1')

    
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    # model, optimizer, lr_scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    model,router, optimizer, lr_scheduler, train_loader, val_loader, ahead_val_loader = accelerator.prepare(model,router, optimizer, lr_scheduler, train_loader, val_loader,ahead_val_loader)
    model.to(device)

    if freze_emb:
        for para in model.embeddings.parameters():
            para.requires_grad = False
    
    for epoch in range(num_epochs):
        
        model.train()
        
        """train origin bert (MLM only)"""
        losses = []
        routes_history = [0 for i in range(2**12)]
        for i, batch in enumerate(train_loader):
            hidden_states_for_router = []
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            # print(epoch, i)
            # print(next(model.parameters()).device)
            # for key, tensor in batch.items():
            #     print(f"{key} is on {tensor.device}")
            _, _, layer_outputs,_ = router(**batch)
            
               
            hidden_states_for_router.append(router.bert.embeddings(batch['input_ids']))
            hidden_states_for_router = hidden_states_for_router  + layer_outputs[0:-1]
            # print(len(hidden_states_for_router))
            
            loss, _, _,_,ids = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers, hidden_states_for_router)
            # print(layers_o[7].shape)
            IDS = get_routes_ids(ids,config)
            for c in IDS:
                routes_history[c] += 1
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()    
        
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator, device, cluster_centers,router)
        ahead_train = validate(model, ahead_val_loader, accelerator, device, cluster_centers,router)
        routes_history = torch.tensor(routes_history)
        listss = torch.argsort(routes_history,descending=True)[:10]
        # loss_test = validate(model, test_loader, accelerator)
        # accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Test Loss: {loss_test}')
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Ahead Valid Loss: {ahead_train}, used-most-10routes: {listss}')

        if accelerator.is_local_main_process:
            writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar('perplexity_valid', loss_valid, epoch)
            writer.add_scalar('perplexity_ahead', ahead_train, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    # accelerator.save_state('./output-formal-1027-new_model-stage1-freeze_embed')
    torch.save(model,'1120_rose_model_satge1.pth')
    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/new_model.json')
    # dataset = RestaurantForLM_small(config=config)
    dataset = Mixdata_1115(config = config)
    ahead_dataset = MixedData(config = config)
    
    device = torch.device("cuda")
    model = base_models.rose_model(config=config)
    router = base_models.BertWithSavers(config=config)
    router.to(device)
    model.to(device)
    # model = nn.DataParallel(model)

    
    train(model=model, num_epochs=50, dataset=dataset, device=device, ahead_dataset = ahead_dataset, router = router)