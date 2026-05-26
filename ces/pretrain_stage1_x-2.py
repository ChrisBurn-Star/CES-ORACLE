import torch.nn as nn
import os
import base_models
from transformers import BertConfig
from Dataset_new import ACLForLM_small, RestaurantForLM_small
from Dataset_new import Wikitext
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from transformer.Transformer import MemoryFromDecoder
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


def validate(model, val_loader, accelerator):
    losses = []
    model.eval()
    for i, batch in enumerate(val_loader):        
        with torch.no_grad():
            batch.to('cuda')
            loss, _, _ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity


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


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    return layer_data


def load_to_cpu(batch):
    batch = {key: tensor.to('cpu') for key, tensor in batch.items()}
    return batch

def load_to_gpu(batch):
    batch = {key: tensor.to('cuda') for key, tensor in batch.items()}
    return batch

def train(model, num_epochs, dataset, dataset_pre):

    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    pre_test_loader, pre_train_loader = dataset_pre.val_loader, dataset_pre.train_loader
    num_updates = num_epochs * len(train_loader)
    # model = torch.load('./output-formal-1/pytorch_model.bin')
    model = torch.load('./output-origin/1014.pth')
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter("log-1019-pcaall/" + 'bert')
    
    
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    # accelerator.load_state("./output-formal-1")
    # accelerator.load_state("./output-formal-2-X-2")
    
    standard_pcas = load_layer_data('layer_pcasXPCA.pth')
    standard_pcas = [data.requires_grad_(True) for data in standard_pcas]
    layer_inputs = load_layer_data('layer_inputsXPCA.pth')
    layer_labels = load_layer_data('layer_labelsXPCA.pth')
    layer_attns = load_layer_data('layer_attnsXPCA.pth')
    standard_pcas_ids = load_layer_data('layer_pcas_idsXPCA.pth')
    #
    standard_scores = load_layer_data('layer_scoresXPCA.pth')
    standard_scores = [data.requires_grad_(True) for data in standard_scores]

    print(len(standard_pcas), len(layer_attns))
    
    
    # model.to('cuda')
    for epoch in range(num_epochs):
        model.train()
        epochp = 0
        
        """train origin bert (MLM only)"""
        losses = []
        for i, batch in enumerate(train_loader):    
            # batch = load_to_gpu(batch)
            # batch_old = {'input_ids': layer_inputs[epoch], 'attention_mask': layer_attns[epoch], 'labels': layer_labels[epoch]}
            # batch = {'input_ids': torch.cat([batch['input_ids'], layer_inputs[0]],0), 'attention_mask': torch.cat([batch['attention_mask'], layer_attns[0]],0), 'labels': torch.cat([batch['labels'], layer_labels[0]],0)}
            # print(batch['input_ids'].shape)
            # loss 1 
            loss, _, _, = model(**batch)
            # loss2, _, _, = model(**batch_old)
            # batch = load_to_cpu(batch)
            # losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            optimizer.zero_grad()
            accelerator.backward(loss)
            # accelerator.backward(loss2)
            # loss.backward()
            optimizer.step()
            lr_scheduler.step()   
            torch.cuda.empty_cache()
            # loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
            # accelerator.print(f'Epoch:{epoch} ({i} Updates)')
            memory = MemoryFromDecoder()
            # loss 2 
            mse_loss = nn.MSELoss()
            batch_old = {'input_ids': layer_inputs[epochp], 'attention_mask': layer_attns[epochp], 'labels': layer_labels[epochp]}

            
            # batch_old = load_to_gpu(batch_old)
            _, _, layer_outputs = model(**batch_old)
            # print(scores0.shape)
            # batch_old = load_to_cpu(batch_old)
            # _, _, layer_outputs = model(**batch_old)
            detached_outputs = [output.detach() for output in layer_outputs]
            detached_outputs = [output.requires_grad_(True) for output in detached_outputs]
            for j, (detached_output, standard_pca) in enumerate(zip(detached_outputs, standard_pcas)):                
                if j % 3 == 0:
                    ids = standard_pcas_ids[j][epochp]
                    # print(detached_output[ids].shape)
                    pca_loss = mse_loss(detached_output[ids], standard_pca[epochp * config.batch_size : (epochp+1) * config.batch_size][ids])                      
                    # print(f'output shape:{detached_output.shape}')
                    # print(pca_loss)
                    local_optimizer = optim.AdamW(model.bert.layers.layers[j].parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6) 
                    # accelerator.print(f'pca_loss:{pca_loss}')
                    local_optimizer.zero_grad()
                    pca_loss.backward(retain_graph=True)

                    local_optimizer.step()
                    torch.cuda.empty_cache()
                    

            # loss 3

            # pca_old = standard_pcas[-1][epochp * config.batch_size : (epochp+1) * config.batch_size]
            # scores = model.head(pca_old)
            # # scores.to('cuda')
            # # batch_old = load_to_cpu(batch_old)
            # detached_scores = scores.detach()
            # # scores.to('cpu')
            # detached_scores = detached_scores.requires_grad_(True)
            
            # # print(detached_scores)
            # # standard_score = standard_scores[epoch]
            # # standard_score = torch.tensor(standard_score)
            # # standard_score.requires_grad_(True)
            # # print([en == sn for en,sn in zip(detached_scores,standard_score)])
            # sc_loss = nn.CrossEntropyLoss()
            # # print(pca_old.shape)
            # score_loss = sc_loss(detached_scores.view(-1, config.vocab_size), batch_old['labels'].view(-1))
            # # print(len())
            # # score_loss = mse_loss(detached_scores, standard_score)
            # # accelerator.print(f'scores_loss:{score_loss}')
            # local_optimizerX = optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999],eps=1e-6)
            # local_optimizerX.zero_grad()
            # score_loss.backward(retain_graph=True)
            # local_optimizerX.step()

            # # del scores, detached_scores, score_loss, pca_loss, loss, local_optimizer, memory, layer_outputs, detached_outputs, detached_output, batch, batch_old

            # torch.cuda.empty_cache()

             # # loss 3
            # # batch_old = {'input_ids': layer_inputs[epoch], 'attention_mask': layer_attns[epoch], 'labels': layer_labels[epoch]}
            

            
            # batch_old = {'input_ids': layer_inputs[epoch], 'attention_mask': layer_attns[epoch], 'labels': layer_labels[epoch]}
            # # batch_old = load_to_gpu(batch_old)
            # _, scores, _ = model(**batch_old)
            # # scores.to('cuda')
            # # batch_old = load_to_cpu(batch_old)
            # detached_scores = scores.detach()
            # # scores.to('cpu')
            # detached_scores = detached_scores.requires_grad_(True)
            # detached_scores = memory.output2input(detached_scores)
            # # print(detached_scores)
            # standard_score = standard_scores[epoch]
            # standard_score = torch.tensor(standard_score)
            # standard_score.requires_grad_(True)
            # # print([en == sn for en,sn in zip(detached_scores,standard_score)])
            # sc_loss = nn.CrossEntropyLoss()
            # score_loss = mse_loss(detached_scores, standard_score)
            # # accelerator.print(f'scores_loss:{score_loss}')
            # local_optimizerX = optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999],eps=1e-6)
            # local_optimizerX.zero_grad()
            # score_loss.backward(retain_graph=True)
            # local_optimizerX.step()

            # del scores, detached_scores, standard_score, score_loss, pca_loss, loss, local_optimizer, memory, layer_outputs, detached_outputs, detached_output, batch, batch_old

            # torch.cuda.empty_cache()

        loss_valid = validate(model, val_loader, accelerator)
        loss_test = validate(model, pre_test_loader, accelerator)
        # accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, pre_Test Loss: {loss_test}')
        accelerator.print(f'Epoch:{epoch} ({i} Updates),  Valid Loss: {loss_valid}, pre_Test Loss: {loss_test}')

        if accelerator.is_local_main_process:
            # writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar('perplexity_valid', loss_valid, epoch)
            writer.add_scalar('perplexity_test', loss_test, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    accelerator.save_state('./output-formal-2-X-1019-pcaall')
    torch.save(model, './output-increamental/1019-1019-pcaall.pth')
    

if __name__ == "__main__":
    set_seed(45)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    config = BertConfig.from_json_file('config/bert.json')
    # dataset = RestaurantForLM(config=config)
    dataset = ACLForLM_small(config=config)
    dataset_pre = RestaurantForLM_small(config=config)
    
    model = base_models.BertWithSavers(config=config)
    # model.to('cuda')
    # model = base_models.BertWithDecoders(config=config)
    # model = nn.DataParallel(model)
    
    train(model=model, num_epochs=70, dataset=dataset, dataset_pre=dataset_pre)