import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import RestaurantForLM_small, MixedData,MixedData_stage1
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim

import torch
import numpy as np
import random

import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import RestaurantForLM_small, ACLForLM_small, MixedData
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch.optim as optim
from transformer.Transformer import MemoryFromDecoder
import torch
import numpy as np
import random
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt



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


def validate(model, val_loader, accelerator, device, centers,pca):
    losses = []
    for i, batch in enumerate(val_loader):  
        batch = {key: tensor.to(device) for key, tensor in batch.items()}      
        with torch.no_grad():
            loss, _, _ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], centers, pca)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity

def get_PCA_obj(inputs, threshold = 0.9):
    # embeddings = [N, embed_dim]
    print('create PCA object')
    inputs = inputs.mean(axis = 1)
    inputs = inputs.cpu().numpy()
    pca = PCA()
    pca.fit(inputs)

    # find the first k components that in total explain threshold of the variance
    unique_dims = 0
    total = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        total += var
        if total >= threshold:
            unique_dims = i
            break

    # print(f'pca unique dims: {unique_dims},  {pca.explained_variance_ratio_[:unique_dims]}')
    # print(torch.tensor(pca.transform(inputs)[:,:unique_dims]).shape)
    return torch.tensor(pca.transform(inputs)[:,:unique_dims]), pca, unique_dims


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    return layer_data
def get_cluster_centers(input,k = 5):
    X = input.cpu().numpy()
    kmeans = KMeans(n_clusters=k, n_init= 'auto')
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers)
    return cluster_centers
def get_gradient_norms(model):
    """Utility function to get gradient norms of a model."""
    return [param.grad.norm().item() for param in model.parameters() if param.grad is not None]
def layer_pca(model, dataset, new_dataset):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader2, val_loader2 = new_dataset.train_loader, new_dataset.val_loader
    num_updates = 70 * len(train_loader)
    model = torch.load('./output-origin/1027.pth')
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()
    
    # load model checkpoint
    model, optimizer, lr_scheduler, train_loader, val_loader, train_loader2, val_loader2 = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, train_loader2, val_loader2)
    # accelerator.load_state(load_path)
    
    # run once
    model.eval()

    out_for_cluster = [[] for i in range(12)]
    cluster_centers = []
    out_for_cluster_special = [[] for i in range(12)]
    special_cluster_centers = []
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i %10 == 0:          
                # print(f"######{i}")                
                _, _, layer_outputs = model(**batch)
                #
               
                out_for_cluster[0].append(model.bert.embeddings(batch['input_ids']))
                out_for_cluster_special[0].append(model.bert.embeddings(batch['input_ids']))
                
                # scores.to('cpu')
                for j, layer_output in enumerate(layer_outputs[:-1]):  
                    # layer_output = layer_output.view(config.batch_size,-1)
                    
                    out_for_cluster_special[j+1].append(layer_output)
                    
                    # out_for_cluster[j+1].append(layer_output)
    # for j in range(len(out_for_cluster)):
    #     out_for_cluster[j] = torch.cat(out_for_cluster[j], dim =0 )
    #     # print(out_for_cluster[j].shape)
    # for j in range(len(out_for_cluster)):
    #     cluster_centers.append(get_cluster_centers(out_for_cluster[j], k = config.num_experts))
    for j in range(len(out_for_cluster_special)):
        out_for_cluster_special[j] = torch.cat(out_for_cluster_special[j], dim =0 )
        # print(out_for_cluster_special[j].shape)
    
    for j in range(len(out_for_cluster_special)):
        out_for_cluster_special_with_pca, pca, uni_dim = get_PCA_obj(out_for_cluster_special[j])
    return pca

def train(model, num_epochs, dataset, device, ahead_dataset, pca):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    
    freze_emb = 1 

    model0 = torch.load('./output-origin/1027.pth')
    cluster_centers = load_layer_data('layer_centers.pth')
    special_centers = load_layer_data('special_layer_centers.pth')
    # print(len(cluster_centers), cluster_centers[9].shape)
    model.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    for i in range(3):
        
        model.layers1[i].load_state_dict(model0.bert.layers.layers[i].state_dict())
    # model.head.load_state_dict(model0.head.state_dict())


    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    ahead_train_loader = ahead_dataset.train_loader
    ahead_val_loader = ahead_dataset.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    writer = SummaryWriter("log-1027/" + 'common_unique-stage1-real')

    
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
            loss, _, _ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], special_centers[3], pca)
            # print(layers_o[7].shape)
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()    
            
        
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator, device, special_centers[3],pca)
        ahead_train = validate(model, ahead_val_loader, accelerator, device, special_centers[3],pca)
        # loss_test = validate(model, test_loader, accelerator)
        # accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Test Loss: {loss_test}')
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Ahead Train Loss: {ahead_train}')

        if accelerator.is_local_main_process:
            writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar('perplexity_valid', loss_valid, epoch)
            # writer.add_scalar('perplexity_test', loss_test, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    accelerator.save_state('./common_unique-stage1-real')
    torch.save(model,'common_unique-stage1-real.pth')
    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/new_model.json')
    # dataset = RestaurantForLM_small(config=config)
    dataset = MixedData_stage1(config = config)
    ahead_dataset = MixedData(config = config)
    
    device = torch.device("cuda")
    model = base_models.simple_model(config=config)
    model.to(device)
    # model = nn.DataParallel(model)
    pca = layer_pca(model = base_models.BertWithSavers(config), dataset=MixedData(config), new_dataset=ACLForLM_small(config))
    
    
    train(model=model, num_epochs=50, dataset=dataset, device=device, ahead_dataset = ahead_dataset, pca = pca)