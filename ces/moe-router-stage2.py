import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import ACLForLM_1103,Mixdata_1115
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
def get_routes_ids(expert_ids,config ):
    d = config.num_experts
    IDS = []
    for i in range(expert_ids[0].shape[0]):
        c = 0
        for j in range(len(expert_ids)):
            c+=expert_ids[j][i]*d**j
        IDS.append(c)
    return IDS

def get_the_other_routes_ids(expert_ids,config):
    IDS = []
    d = config.num_experts
    for i in range(expert_ids[0].shape[0]):
        c = 0
        for j in range(len(expert_ids)):
            if expert_ids[j][i] == 1:
                t = 0
            else:
                t = 1
            c+=t*d**j
        IDS.append(c)
    return IDS


def train(model, num_epochs, dataset, device, ahead_dataset, router):
    path_replay = 1
    other_routes = 0
    REPLAY_STEPS = 20

    model1 = torch.load('1120_rose_model_satge1.pth')
    # cluster_centers = load_layer_data('1115-layer_centers-2expert.pth')
    cluster_centers = load_layer_data('1121-layer_centers-3expert.pth')

    test = load_layer_data('1121-replay_data_rose_models-2experts.pth')
    model.embeddings.load_state_dict(model1.embeddings.state_dict())
    # for i in range(config.num_hidden_layers):
    #     for j in range(config.num_experts-1):
    #         model.layers[i].experts[j].load_state_dict(model1.layers[i].experts[j].state_dict())
    # model.head.load_state_dict(model1.head.state_dict())
    model.load_state_dict(model1.state_dict())
    model0 = torch.load('1115-only-bert-formoe-stage0.pth')
    router.bert.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    for i in range(config.num_hidden_layers):
        router.bert.layers.layers[i].load_state_dict(model0.bert.encoders.layers[i].state_dict())
    router.head.load_state_dict(model0.head.state_dict())


    

    print(len(cluster_centers), cluster_centers[9].shape)



    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    ahead_train_loader, ahead_val_loader = ahead_dataset.train_loader, ahead_dataset.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    writer = SummaryWriter('1120_rose_model_satge2-changecenters-replay')

    
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    model,router, optimizer, lr_scheduler, train_loader, val_loader,ahead_val_loader = accelerator.prepare(model,router, optimizer, lr_scheduler, train_loader, val_loader,ahead_val_loader)
    model.to(device)

    for i, batch in enumerate(ahead_train_loader):
        if i == 22:
            batch0 = batch.to('cuda')
    for epoch in range(num_epochs):
        model.train()
        losses = []
        
        routes_history = [0 for i in range(config.num_experts**12)]
        for i, batch in enumerate(train_loader):
            
            # batch = {'input_ids': torch.cat((batch['input_ids'],batch0['input_ids']),dim = 0), 'attention_mask': torch.cat((batch['attention_mask'],batch0['attention_mask']),dim = 0), 'labels': torch.cat((batch['labels'],batch0['labels']),dim = 0)}
            # batch = batch0
            hidden_states_for_router = []
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            # print(epoch, i)
            # print(next(model.parameters()).device)
            # for key, tensor in batch.items():
            #     print(f"{key} is on {tensor.device}")
            _, _, layer_outputs,_ = router(**batch)
                #
               
            hidden_states_for_router.append(router.bert.embeddings(batch['input_ids']))
            hidden_states_for_router = hidden_states_for_router  + layer_outputs[0:-1]
            # print(epoch, i)
            loss, _, _ ,_, EXPERT_IDS= model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers, hidden_states_for_router)
            
            
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            # print(EXPERT_IDS)



            IDS = get_routes_ids(EXPERT_IDS,config)
            for c in IDS:
                routes_history[c] += 1
            other_IDS = get_the_other_routes_ids(EXPERT_IDS,config)
            # print(IDS)
            if path_replay:
                if i%REPLAY_STEPS == 0:
                    for c in IDS:
                        if len(test[c]):
                            # print(f'epoch:{epoch}, i:{i}, replay path:{c}')
                            routes_history[c] += 1
                            if len(test[c])>=10:
                                kts =10
                            else:
                                kts = len(test[c])
                            for jii in range(kts):
                                hidden_states_for_routerd = []
                                _, _, layer_outputsd,_ = router(test[c][jii]['input_ids'].view(1,config.seq_len),test[c][jii]['attention_mask'].view(1,config.seq_len), test[c][jii]['labels'].view(1,config.seq_len))
                                    #
                                
                                hidden_states_for_routerd.append(router.bert.embeddings(batch['input_ids']))
                                hidden_states_for_routerd = hidden_states_for_routerd  + layer_outputsd[0:-1]
                                replay_loss, _, _ ,_, _= model(test[c][jii]['input_ids'].view(1,config.seq_len),test[c][jii]['attention_mask'].view(1,config.seq_len), test[c][jii]['labels'].view(1,config.seq_len), cluster_centers, hidden_states_for_routerd)
                                optimizer.zero_grad()
                                accelerator.backward(replay_loss)
                                optimizer.step()
                    if other_routes:
                        for c in other_IDS:
                            if len(test[c]):
                                # print(f'epoch:{epoch}, i:{i}, replay path:{c}')
                                routes_history[c] += 1
                            if len(test[c])>=10:
                                kts =10
                            else:
                                kts = len(test[c])
                            for jii in range(kts):
                                    # print(test[c][jii]['input_ids'].shape)
                                    replay_loss, _, _ ,_, _= model(test[c][jii]['input_ids'].view(1,config.seq_len),test[c][jii]['attention_mask'].view(1,config.seq_len), test[c][jii]['labels'].view(1,config.seq_len), cluster_centers)
                                    optimizer.zero_grad()
                                    accelerator.backward(replay_loss)
                                    optimizer.step()
                
            else:
                pass
        
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator, device, cluster_centers, router)
        ahead_train = validate(model, ahead_val_loader, accelerator, device, cluster_centers, router)
        routes_history = torch.tensor(routes_history)
        listss = torch.argsort(routes_history,descending=True)[:10]
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Ahead Train Loss: {ahead_train}, used-most-10routes: {listss}')

        if accelerator.is_local_main_process:
            writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar('perplexity_valid', loss_valid, epoch)
            writer.add_scalar('perplexity_ahead', ahead_train, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
    # torch.save(model,'1120_rose_model_satge2-changecenters.pth')
    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/new_model.json')
    dataset = ACLForLM_1103(config = config)
    ahead_dataset = Mixdata_1115(config = config)
    device = torch.device("cuda")
    model = base_models.rose_model(config=config)
    router = base_models.BertWithSavers(config=config)
    router.to(device)
    model.to(device)

    train(model=model, num_epochs=50, dataset=dataset, device=device, ahead_dataset = ahead_dataset, router = router)