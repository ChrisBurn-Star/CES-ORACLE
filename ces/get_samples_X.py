import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import RestaurantForLM_small
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


def get_gradient_norms(model):
    """Utility function to get gradient norms of a model."""
    return [param.grad.norm().item() for param in model.parameters() if param.grad is not None]



def differentiable_pca(x, k=2):
    scaler = StandardScaler()
    standarlized_x = scaler.fit_transform(x.cpu().numpy())
    x = torch.from_numpy(standarlized_x)
    x = x.to('cuda')
    # Perform SVD
    U, S, V = torch.svd(x)

    # Extract the top k principal components
    principal_components = U[:, :k]

    # Project data onto these components
    reduced_data = x @ V[:, :k]

    return reduced_data


def get_outputs_sample(output, k =2):

    kmeans = KMeans(n_clusters=k, n_init= 'auto')
    
    dbscan = DBSCAN(eps=5, min_samples=2)
    ypred = dbscan.fit_predict(output.cpu().numpy())
    print(max(ypred))
    # Fit the model to your data
    kmeans.fit(output.cpu().numpy())

    # Get cluster assignments for each data point
    labels = kmeans.labels_

    # Get the coordinates of the cluster centers
    cluster_centers = kmeans.cluster_centers_

    return torch.tensor(cluster_centers)

def get_new_output(output):
    ase = torch.zeros(output.shape[0],output.shape[1])
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            ase[i][j] = torch.mean(output[i][j])

    return ase

def layer_pca(model, dataset, load_path):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    num_updates = 70 * len(train_loader)
    model = torch.load('./output-origin/1014.pth')
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()
    
    # load model checkpoint
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    # accelerator.load_state(load_path)
    
    # run once
    model.eval()
    
    all_layer_outputs = [[] for i in range(12)]
    all_layer_inputs = []
    all_layer_labels = []
    all_layer_attns = []
    #
    all_scores_len = []
    memory = MemoryFromDecoder()
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i % 8 == 0:                          
                _, scores, layer_outputs = model(**batch)
                #
                scores = memory.output2input(scores)
                # scores.to('cpu')
                for j, layer_output in enumerate(layer_outputs):  
                    # layer_output = layer_output.view(config.batch_size,-1)
                    all_layer_outputs[j].append(layer_output)
                input = batch['input_ids']
                label = batch['labels']
                attention_mask = batch['attention_mask']                    
                all_layer_inputs.append(input)
                all_layer_labels.append(label)
                all_layer_attns.append(attention_mask)
                all_scores_len.append(scores)
    
    accelerator.print(f'Number of Samples batches: {len(all_layer_outputs[0])}')
    
    # calculate pca
    layer_pcas = {}
    layer_inputs = {}
    layer_labels = {}
    layer_attns = {}
    #
    layer_scores = {}

    scaler = StandardScaler()
    for i, layer in enumerate(all_layer_outputs):
        layer_np = [single.cpu().numpy() for single in layer]
        layer = np.vstack(layer_np)
        layer = torch.from_numpy(layer)        

        layer_pcas['layer ' + str(i+1) ] = layer
        print(layer.size())
        
    for i, layer in enumerate(all_layer_inputs):
        layer_inputs['layer' + str(i+1) ] = all_layer_inputs[i]
        layer_labels['layer' + str(i+1) ] = all_layer_labels[i]
        layer_attns['layer' + str(i+1)] = all_layer_attns[i]
    #
    for i, layer in enumerate(all_scores_len):
        print(layer.size())
        layer_scores['layer' + str(i+1) ] = all_scores_len[i]
    print(layer_scores.keys())
    print(all_scores_len[0])
    # save pcas
    torch.save(layer_pcas, 'layer_pcasX.pth')
    torch.save(layer_inputs, 'layer_inputsX.pth')
    torch.save(layer_labels, 'layer_labelsX.pth')
    torch.save(layer_attns, 'layer_attnsX.pth')
    #
    torch.save(layer_scores, 'layer_scoresX.pth')


if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/bert.json')
    dataset = RestaurantForLM_small(config=config)
    
    model = base_models.BertWithSavers(config=config)
    # model = base_models.BertWithDecoders(config=config)
    # model = nn.DataParallel(model)
    
    load_path = "./output-formal-1P"
    layer_pca(model=model, dataset=dataset, load_path=load_path)
