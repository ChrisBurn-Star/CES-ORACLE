import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import RestaurantForLM_small, MixedData,MixedData_stage1,old_MixedData_after_stage1, ACLForLM
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


def validate(model, val_loader, accelerator, device, centers):
    losses = []
    for i, batch in enumerate(val_loader):  
        batch = {key: tensor.to(device) for key, tensor in batch.items()}      
        with torch.no_grad():
            loss, _, _ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], centers)
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


def train(model, num_epochs, dataset, device, ahead_dataset):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    
    freze_emb = 1 
    model = torch.load('common_unique-stage1.pth')
    # model0 = torch.load('./output-origin/1027.pth')
    # cluster_centers = load_layer_data('layer_centers.pth')
    special_centers = load_layer_data('special_layer_centers.pth')
    # print(len(cluster_centers), cluster_centers[9].shape)
    # model.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    # for i in range(3):
        
    #     model.layers1[i].load_state_dict(model0.bert.layers.layers[i].state_dict())
    # # model.head.load_state_dict(model0.head.state_dict())


    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    ahead_train_loader = ahead_dataset.train_loader
    ahead_val_loader = ahead_dataset.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    writer = SummaryWriter("log-1027/" + 'common_unique-stage2')

    
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    # model, optimizer, lr_scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    model.to(device)

    if freze_emb:
        for para in model.embeddings.parameters():
            para.requires_grad = False
        for para in model.layers1[0:3].parameters():
            para.requires_grad = False
    
    for epoch in range(num_epochs):
        
        model.train()
        
        """train origin bert (MLM only)"""
        losses = []
        for i, batch in enumerate(train_loader):
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            # print(epoch, i)
            # print(next(model.parameters()).device)
            # for key, tensor in batch.items():
            #     print(f"{key} is on {tensor.device}")
            loss, _, _ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], special_centers[3])
            # print(layers_o[7].shape)
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()    
            
        
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator, device, special_centers[3])
        ahead_train = validate(model, ahead_val_loader, accelerator, device, special_centers[3])
        # loss_test = validate(model, test_loader, accelerator)
        # accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Test Loss: {loss_test}')
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Ahead Train Loss: {ahead_train}')

        if accelerator.is_local_main_process:
            writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar('perplexity_valid', loss_valid, epoch)
            # writer.add_scalar('perplexity_test', loss_test, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    accelerator.save_state('./common_unique-stage2')
    torch.save(model,'common_unique-stage2.pth')
    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/new_model.json')
    # dataset = RestaurantForLM_small(config=config)
    dataset = ACLForLM(config = config)
    ahead_dataset = old_MixedData_after_stage1(config = config)
    
    device = torch.device("cuda")
    model = base_models.simple_model(config=config)
    model.to(device)
    # model = nn.DataParallel(model)

    
    train(model=model, num_epochs=50, dataset=dataset, device=device, ahead_dataset = ahead_dataset)