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
    model = torch.load('./output-origin/1019.pth')
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()


    
    use_pca =1
    replay_12 = 1
    replay_decoder =0
    all_thorugh =1 
    decoder_self = 1
    replay_12_through =0
    replay_data =0
    add_loss = 0
    decoder_self_g = 4
    decoder_self_k = 5
    add_loss_k = 1
    test = 4
    

    
    epochp = 0

    ck = 1

    RG = True

    print('replay_12_through*use_pca == 1 must!!! ')
    print(f'12 layers:{replay_12}; decoder_replay:{replay_decoder}; tube between layers:{replay_12_through}; data_replay:{replay_data}; use pca:{use_pca}; test:{test}; all_through:{all_thorugh}; decoder_self:{decoder_self}； add_loss:{add_loss}; decoder_self_k:{decoder_self_k}: decoder_self_g:{decoder_self_g}')
    file_name0 = "logs/"+'CK1000-output-12_layers_%d-decoder_replay%d-tube_between_layers%d-data_replay%d-use_pca%d-test%d-all_through%d-decoder_self%d-add_loss%d-decoder_self_k%d-decoder_self_g%d'%(replay_12, replay_decoder, replay_12_through, replay_data, use_pca,test,all_thorugh, decoder_self,add_loss,decoder_self_k, decoder_self_g)
    writer = SummaryWriter(file_name0)
    
    
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
    sc_loss = nn.CrossEntropyLoss()

    
    # model.to('cuda')
    for epoch in range(num_epochs):
        model.train()
        
        
        """train origin bert (MLM only)"""
        losses = []
        for i, batch in enumerate(train_loader):    
            # batch = load_to_gpu(batch)
            # batch_old = {'input_ids': layer_inputs[epoch], 'attention_mask': layer_attns[epoch], 'labels': layer_labels[epoch]}
            if replay_data:
                batch = {'input_ids': torch.cat([batch['input_ids'], layer_inputs[0]],0), 'attention_mask': torch.cat([batch['attention_mask'], layer_attns[0]],0), 'labels': torch.cat([batch['labels'], layer_labels[0]],0)}
            else:
                pass
            # print(batch['input_ids'].shape)
            # loss 1 
            loss, _, _, = model(**batch)
            # loss2, _, _, = model(**batch_old)
            # batch = load_to_cpu(batch)
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            optimizer.zero_grad()
            accelerator.backward(loss)
            # accelerator.backward(loss2)
            # loss.backward()
            
            loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
            # accelerator.print(f'Epoch:{epoch} ({i} Updates)')
            memory = MemoryFromDecoder()
            # loss 2 
            if replay_12:

                mse_loss = nn.MSELoss()
                batch_old = {'input_ids': layer_inputs[epochp], 'attention_mask': layer_attns[epochp], 'labels': layer_labels[epochp]}
                batch_old = load_to_gpu(batch_old)
                _, _, layer_outputs = model(**batch_old)
                inputs = model.bert.embeddings(batch_old['input_ids'])

                # print(scores0.shape)
                # batch_old = load_to_cpu(batch_old)
                # _, _, layer_outputs = model(**batch_old)
                detached_outputs = [output.detach() for output in layer_outputs]
                detached_outputs = [output.requires_grad_(True) for output in detached_outputs]
                for j, (detached_output, standard_pca) in enumerate(zip(detached_outputs, standard_pcas)):
                    local_optimizer = optim.AdamW(model.bert.layers.layers[j].parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
                    local_optimizer.zero_grad()
                    ids = standard_pcas_ids[j][epochp]
                    # print(detached_output[ids].shape)
                    # pca_loss = mse_loss(detached_output[ids], standard_pca[epochp * config.batch_size : (epochp+1) * config.batch_size][ids]) 
                    if j ==0:
                        inputs =inputs
                    else:
                        inputs = standard_pcas[j-1][epochp * config.batch_size : (epochp+1) * config.batch_size]
                    
                    if use_pca:
                        detached_output = detached_output[ids]
                        standard_pca = standard_pca[epochp * config.batch_size : (epochp+1) * config.batch_size][ids]
                        inputs = inputs[ids]
                        labels = batch_old['labels'][ids]
                        attns = batch_old['attention_mask'][ids]
                    else:
                        detached_output = detached_output
                        standard_pca = standard_pca[epochp * config.batch_size : (epochp+1) * config.batch_size]
                        inputs = inputs
                        labels = batch_old['labels']
                        attns = batch_old['attention_mask']
                    layer_output = model.bert.layers.layers[j](inputs, attns).detach()
                    layer_output.requires_grad_(True)
                    # layer_output = model.bert.layers.layers[j](inputs, attns)

                    replay_layer_loss = mse_loss(layer_output, standard_pca)
                    add_loss = replay_layer_loss
                    if i == 300:
                        accelerator.print(f'replay_layer_loss:{replay_layer_loss}') 
                    replay_layer_loss = replay_layer_loss * ck
                    
                    replay_layer_loss.backward(retain_graph=RG)
                    
                    if all_thorugh:
                        det = inputs
                        for k in range(j,12):
                            det = model.bert.layers.layers[k](det, attns)
                        scores_for_replay = model.head(det).detach()
                        scores_for_replay.requires_grad_(True)
                        # scores_for_replay = model.head(det)

                        relay_loss_from_decoder  = sc_loss(scores_for_replay.view(-1, config.vocab_size), labels.view(-1))
                        add_loss = add_loss + relay_loss_from_decoder
                    else:
                        pass
                    
                    
                    if replay_12_through:
                        det = inputs
                        for k in range(j,12):
                            det = model.bert.layers.layers[k](det, attns)
                            layer_out = det.detach()
                            layer_out.requires_grad_(True)
                            # layer_out = det
                            if use_pca:
                                replay_layer_loss_sub = mse_loss(layer_out, standard_pcas[k][epochp * config.batch_size : (epochp+1) * config.batch_size][ids])
                            else:
                                replay_layer_loss_sub = mse_loss(layer_out, standard_pcas[k][epochp * config.batch_size : (epochp+1) * config.batch_size])
                            local_optimizerK = optim.AdamW(model.bert.layers.layers[k].parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6) 
                            local_optimizerK.zero_grad()
                            replay_layer_loss_sub = replay_layer_loss_sub * ck
                            add_loss = add_loss + replay_layer_loss_sub
                            replay_layer_loss_sub.backward(retain_graph=RG)
                            local_optimizerK.step()
                    else:
                        pass
    
                    
                    

                    if all_thorugh:
                        # relay_loss_from_decoder.backward(retain_graph=RG)
                        relay_loss_from_decoder = relay_loss_from_decoder * ck
                        if decoder_self:
                            if (i*decoder_self_g) %decoder_self_k == 0:

                                local_optimizerD = optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6) 
                                # accelerator.print(f'pca_loss:{pca_loss}')
                                local_optimizerD.zero_grad()

                                relay_loss_from_decoder.backward(retain_graph=RG)
                                local_optimizerD.step()
                        else:
                            relay_loss_from_decoder.backward(retain_graph=RG)
                    
                    local_optimizer.step()
                    
                    
                    torch.cuda.empty_cache()
            if add_loss:
                accelerator.backward(add_loss * add_loss_k)
            optimizer.step()
            lr_scheduler.step()   
            torch.cuda.empty_cache()
            # loss 3
            if replay_decoder:

                pca_old = standard_pcas[-1][epochp * config.batch_size : (epochp+1) * config.batch_size]
                scores = model.head(pca_old)
                # scores.to('cuda')
                # batch_old = load_to_cpu(batch_old)
                detached_scores = scores.detach()
                # scores.to('cpu')
                detached_scores = detached_scores.requires_grad_(True)
                
                # print(detached_scores)
                # standard_score = standard_scores[epoch]
                # standard_score = torch.tensor(standard_score)
                # standard_score.requires_grad_(True)
                # print([en == sn for en,sn in zip(detached_scores,standard_score)])
                
                # print(pca_old.shape)
                score_loss = sc_loss(detached_scores.view(-1, config.vocab_size), batch_old['labels'].view(-1))
                # print(len())
                # score_loss = mse_loss(detached_scores, standard_score)
                # accelerator.print(f'scores_loss:{score_loss}')
                local_optimizerX = optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999],eps=1e-6)
                local_optimizerX.zero_grad()
                score_loss.backward(retain_graph=RG)
                local_optimizerX.step()

                # del scores, detached_scores, score_loss, loss, memory, batch, batch_old

                torch.cuda.empty_cache()

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
            # score_loss.backward(retain_graph=RG)
            # local_optimizerX.step()

            # del scores, detached_scores, standard_score, score_loss, pca_loss, loss, local_optimizer, memory, layer_outputs, detached_outputs, detached_output, batch, batch_old

            # torch.cuda.empty_cache()

        loss_valid = validate(model, val_loader, accelerator)
        loss_test = validate(model, pre_test_loader, accelerator)
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, pre_Test Loss: {loss_test}')
        accelerator.print(f'Epoch:{epoch} ({i} Updates),  Valid Loss: {loss_valid}, pre_Test Loss: {loss_test}')

        if accelerator.is_local_main_process:
            # writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar('perplexity_valid', loss_valid, epoch)
            writer.add_scalar('perplexity_test', loss_test, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
    file_name = './output-'+'CK1000-12_layers_%d-decoder_replay%d-tube_between_layers%d-data_replay%d-use_pca%d-test%d-all_through%d-decoder_self%d-add_loss%d-decoder_self_k%d-decoder_self_g%d'%(replay_12, replay_decoder, replay_12_through, replay_data, use_pca,test,all_thorugh, decoder_self, add_loss, decoder_self_k, decoder_self_g)
    accelerator.save_state(file_name)
    torch.save(model, file_name+'.pth')
    

def eval(model, dataset):
    accelerator = Accelerator()
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
    #test_loader = dataset.test_loader

    model, val_loader = accelerator.prepare(model, val_loader)
    
    PPL_test = validate(model, train_loader, accelerator)
    accelerator.print(f'PPL TEST: {PPL_test}')


if __name__ == "__main__":
    set_seed(45)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    config = BertConfig.from_json_file('config/bert.json')
    # dataset = RestaurantForLM(config=config)
    dataset = ACLForLM_small(config=config)
    dataset_pre = RestaurantForLM_small(config=config)
    
    model = base_models.BertWithSavers(config=config)
    model.to('cuda')
    # model = base_models.BertWithDecoders(config=config)
    # model = nn.DataParallel(model)
    # model = torch.load('./output-12_layers_0-decoder_replay0-tube_between_layers0-data_replay0-use_pca0.pth')
    # eval(model, dataset_pre)
    
    train(model=model, num_epochs=50, dataset=dataset, dataset_pre=dataset_pre)