import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import RestaurantForLM_small, MixedData,MixedData_stage1, ACLForLM, old_MixedData_after_stage1,ACLForLM_1103
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
            loss, _, _,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], centers)
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
    
    freze_emb = 0
    replay_datao = 0
    decoder_replay = 0
    add_loss = 0
    path_replay = 1
    REPLAY_STEPS = 20
    # print(f'freze_emb{freze_emb}  replay_datao{replay_datao}  decoder_replay{decoder_replay}  add_loss{add_loss} ')

    if freze_emb:
        model = torch.load('1027-stage1-new_model-freeze_embed.pth')
    else:
        model = torch.load('1102-stage1-new_model-2-experts.pth')
        # model = torch.load('1027-stage1-new_model.pth')
    # cluster_centers = load_layer_data('layer_centers_after_stage1.pth')
    cluster_centers = load_layer_data('layer_centers-2expert.pth')
    # cluster_centers = load_layer_data('layer_centers.pth')

    replay_outputs = load_layer_data('layer_replays_new_models-2experts.pth')
    replay_inputs = load_layer_data('input_layer_replays_new_models-2experts.pth')
    replay_attens = load_layer_data('atten_layer_replays_new_models-2experts.pth')
    replay_label = load_layer_data('label_layer_replays_new_models-2experts.pth')
    test = load_layer_data('replay_data_new_models-2experts.pth')
    

    # vaild_replay_data = [[[]for j in range(5)] for i in range(12)]

    print(len(cluster_centers), cluster_centers[9].shape)
    # model.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    # for i in range(config.num_hidden_layers):
    #     for j in range(config.num_experts):
    #         model.layers[i].experts[j].load_state_dict(model0.bert.layers.layers[i].state_dict())
    # model.head.load_state_dict(model0.head.state_dict())


    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    ahead_train_loader, ahead_val_loader = ahead_dataset.train_loader, ahead_dataset.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    writer = SummaryWriter("log-1102/" + '1115-32-new_model-pathreplay-2experts')

    
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    # model, optimizer, lr_scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    model.to(device)
    if freze_emb:
        for para in model.embeddings.parameters():
            para.requires_grad = False
        cluster_centers = load_layer_data('layer_centers.pth')
    for i, batch in enumerate(ahead_train_loader):
        if i == 22:
            batch0 = batch.to('cuda')
    for epoch in range(num_epochs):
        
        model.train()
        
        """train origin bert (MLM only)"""
        losses = []
        for i, batch in enumerate(train_loader):
            if i < 600:
                # print(i)
                # batch = {'input_ids': torch.cat((batch['input_ids'],batch0['input_ids']),dim = 0), 'attention_mask': torch.cat((batch['attention_mask'],batch0['attention_mask']),dim = 0), 'labels': torch.cat((batch['labels'],batch0['labels']),dim = 0)}
                # batch = batch0
                batch = {key: tensor.to(device) for key, tensor in batch.items()}
                # print(epoch, i)
                # print(next(model.parameters()).device)
                # for key, tensor in batch.items():
                #     print(f"{key} is on {tensor.device}")
                loss, _, _ ,_, EXPERT_IDS= model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers)
                # print(loss)
                # print(layers_o[7].shape)
                losses.append(accelerator.gather(loss.repeat(config.batch_size)))
                
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()


                # loss, _, _ ,_, _= model(batch0['input_ids'],batch0['attention_mask'], batch0['labels'], cluster_centers)
                # # print(layers_o[7].shape)
                # # losses.append(accelerator.gather(loss.repeat(config.batch_size)))
                
                
                # optimizer.zero_grad()
                # accelerator.backward(loss)
                # optimizer.step()
                lr_scheduler.step()
                # if not add_loss:
                #     optimizer.step()
                #     lr_scheduler.step()
                # else:
                #     pass



                if path_replay:
                    if i%REPLAY_STEPS == 0:
                        c = 0
                        for jk in range(len(EXPERT_IDS)):
                            c += EXPERT_IDS[jk]*2**jk
                        if len(test[c]):
                            # print(f'epoch:{epoch}, i:{i}, replay path:{c}')
                            for jii in range(len(test[c])):
                                replay_loss, _, _ ,_, _= model(test[c][jii]['input_ids'],test[c][jii]['attention_mask'], test[c][jii]['labels'], cluster_centers)
                                optimizer.zero_grad()
                                accelerator.backward(replay_loss)
                                optimizer.step()
                    
                mse_loss = nn.MSELoss()
                ce_loss = nn.CrossEntropyLoss()
                if replay_datao:
                    add_lossr = 0.0
                    for it in range(config.num_hidden_layers):
                        expert_id = EXPERT_IDS[it]
                        if (len(replay_inputs[it][expert_id])):
                            new_output = model.layers[it].experts[expert_id](replay_inputs[it][expert_id][:16], replay_attens[it][expert_id][:16])
                            new_output = new_output.detach()
                            new_output.requires_grad_(True)
                            replay_loss = mse_loss(new_output, replay_outputs[it][expert_id][:16])
                            # print(replay_outputs[it][expert_id].shape)
                            local_optimizer = optim.AdamW(model.layers[it].experts[expert_id].parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
                            local_optimizer.zero_grad()
                            replay_loss.backward(retain_graph=True)
                            local_optimizer.step()
                            if add_loss:
                                add_lossr = add_lossr + replay_loss
                            ############
                            if decoder_replay:
                                for kt in range(it+1, config.num_hidden_layers):
                                    expert_idt = EXPERT_IDS[kt]
                                    new_output = model.layers[kt].experts[expert_idt](new_output, replay_attens[it][expert_id])
                                new_output = new_output.detach()
                                new_output.requires_grad_(True)
                                new_score = model.head(new_output)
                                new_loss = ce_loss(new_score.view(-1, config.vocab_size), replay_label[it][expert_id].view(-1))
                                local_optimizerH = optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
                                local_optimizerH.zero_grad()
                                new_loss.backward(retain_graph=True)
                                local_optimizerH.step()
                            #############

                        else:
                            pass
                else:
                    pass

                if add_loss:
                    add_lossr.backward(retain_graph=True)
                    optimizer.step()
                    lr_scheduler.step()
                
                    
            else:
                break   
        
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator, device, cluster_centers)
        ahead_train = validate(model, ahead_val_loader, accelerator, device, cluster_centers)
        # loss_test = validate(model, test_loader, accelerator)
        # accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Test Loss: {loss_test}')
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Ahead Train Loss: {ahead_train}')

        if accelerator.is_local_main_process:
            writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar('perplexity_valid', loss_valid, epoch)
            writer.add_scalar('perplexity_ahead', ahead_train, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    # accelerator.save_state('./output-formal-1101-new_model-stage2_replay_lessdata_head')
    # torch.save(model,'1114-32-new_model-steps-replay-raw_data-5experts.pth')
    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/new_model.json')
    # dataset = RestaurantForLM_small(config=config)
    dataset = ACLForLM(config = config)
    ahead_dataset = old_MixedData_after_stage1(config = config)
    
    device = torch.device("cuda")
    model = base_models.new_model(config=config)
    model.to(device)
    # model = nn.DataParallel(model)

    
    train(model=model, num_epochs=50, dataset=dataset, device=device, ahead_dataset = ahead_dataset)