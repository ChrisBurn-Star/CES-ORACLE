import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import MoMoE_MIXED_0128,MoMoE_MIXED_WIKI103_0124,MoMoE_MIXED,MoMoE_WIKI103,MixedData_1211,MixedData_0110_1,MixedData_1211_1,RestaurantForLM_small, MixedData,MixedData_stage1,Mixdata_1103,Mixdata_1115,Wikitxt103ForLM_1103,Wikitxt103ForLM_0102_rose
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


# def validate(model, val_loader, accelerator, device, centers,model0,device2):
#     losses = []
#     for i, batch in enumerate(val_loader):  
#         batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
#         batch = {key: tensor.to(device) for key, tensor in batch.items()}

#         _,_,_,_,_,_,inputs = model0(**batch0)
#         inputs = [i0.to(device) for i0 in inputs]
#         with torch.no_grad():
            
#             loss, _, _, _, _,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], centers,inputs)
#         losses.append(accelerator.gather(loss.repeat(len(batch))))
    
#     losses = torch.cat(losses)[:len(val_loader.dataset)]
#     perplexity = torch.mean(losses)
    
#     return perplexity

# def validate(model, val_loader, accelerator, device, centers):
#     losses = []
#     for i, batch in enumerate(val_loader):  

#         batch = {key: tensor.to(device) for key, tensor in batch.items()}
#         with torch.no_grad():
            
#             loss, _, _, _, _,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], centers)
#         losses.append(accelerator.gather(loss.repeat(len(batch))))
    
#     losses = torch.cat(losses)[:len(val_loader.dataset)]
#     perplexity = torch.mean(losses)
    
#     return perplexity


def validate(model, val_loader, accelerator, device):
    losses = []
    model.deval =1
    for i, batch in enumerate(val_loader):  

        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        with torch.no_grad():
            
            loss, _, _, _, _,_,_,_ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    model.deval =0
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

def train(model, num_epochs, dataset, device, ahead_dataset):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    
    freze_emb = 0 
    AngeL = 0
    sparsity_rate = 0
    SPAR = 1
    orth_rate = 0
    ORTH = 1
    same_in_param = 0

    with_bert_param = 1
    # model = torch.load('0119_AngeL_rose_model_4expert_seq128_batchsize64_WIKI103_WP10.pth')
    # model0 = base_models.BertForMLM_toshow(config=config)
    # model01 = torch.load('0119-bert-MIXED-WP3.pth')
    # model0.load_state_dict(model01.state_dict())

    # model0 = torch.load('0119-bert-WIKI103-WP10.pth')
    cluster_centers = load_layer_data('0119-layer_centers-4t-MIXED-WP3.pth')

    # cluster_centers = load_layer_data('0119-layer_centers-4t-WIKI103-WP10.pth')
    
    # print(len(cluster_centers), cluster_centers[9].shape)
    print()
    # if same_in_param:
    #     for i in range(config.num_hidden_layers):
    #         for j in range(1,config.num_experts):
    #             model.layers[i].experts[j].load_state_dict(model.layers[i].experts[0].state_dict())
    # if with_bert_param:
    #     model.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
        # for i in range(config.num_hidden_layers):
        #     for j in range(config.num_experts):
        #         model.layers[i].experts[j].load_state_dict(model0.bert.encoders.layers[i].state_dict())
        # model.head.load_state_dict(model0.head.state_dict())
    # if not AngeL:
    #     model.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    #     for i in range(config.num_hidden_layers):
    #         for j in range(config.num_experts):
    #             model.layers[i].experts[j].load_state_dict(model0.bert.encoders.layers[i].state_dict())
    #     model.head.load_state_dict(model0.head.state_dict())

    # else:
        # model.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
        # for i in range(config.num_hidden_layers):
        #     for j in range(config.num_experts):
                


        #         # model.layers[i].experts[j].attention.load_state_dict(model0.bert.encoders.layers[i].LayerNorm.state_dict())

        #         model.layers[i].experts[j].LayerNorm.load_state_dict(model0.bert.encoders.layers[i].LayerNorm.state_dict())
                
        #         model.layers[i].experts[j].dropout.load_state_dict(model0.bert.encoders.layers[i].dropout.state_dict())
        #         model.layers[i].experts[j].ffn.load_state_dict(model0.bert.encoders.layers[i].ffn.state_dict())
        # model.head.load_state_dict(model0.head.state_dict())
        # pass
    # router.bert.embeddings.load_state_dict(model0.bert.embeddings.state_dict())
    # for i in range(config.num_hidden_layers):
    #     router.bert.layers.layers[i].load_state_dict(model0.bert.encoders.layers[i].state_dict())

    # router.head.load_state_dict(model0.head.state_dict())

    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    # ahead_train_loader = ahead_dataset.train_loader
    # ahead_val_loader = ahead_dataset.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    # writer = SummaryWriter('tensorboard_0119/0120_AngeL_rose_model_4expert_seq128_batchsize64_WIKI103_WP10_LAR3')
    writer = SummaryWriter('tensorboard_0119/0128_AngeL_rose_model_2t_seq128_batchsize64_LAR_autoinit_WP128000')


    
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    # model, optimizer, lr_scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    model, optimizer, lr_scheduler, train_loader, val_loader= accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    model.to(device)
    # device2 = torch.device("cuda:3")
    # model0.to(device2)
    # router.to(device)
    # if freze_emb:
    #     for para in model.embeddings.parameters():
    #         para.requires_grad = False
    steps = 0
    losses = []
    WI_Losses = []
    FN_Losses = []
    for epoch in range(num_epochs):
        
        model.train()
        
        
        # routes_history = [0 for i in range(4**12)]
        for i, batch in enumerate(train_loader):
            steps+= 1
            # hidden_states_for_router = []
            # batch0 = {key: tensor.to(device2) for key, tensor in batch.items()}
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            # print(epoch, i)
            # print(next(model.parameters()).device)
            # for key, tensor in batch.items():
            #     print(f"{key} is on {tensor.device}")
            # _, _, layer_outputs,_ = router(**batch)
            
            
            # _,_,_,_,_,_,inputs = model0(**batch0)
            # hidden_states_for_router.append(router.bert.embeddings(batch['input_ids']).to(device))
            # hidden_states_for_router = hidden_states_for_router  + layer_outputs[0:-1]
            # # print(len(hidden_states_for_router))
            # inputs = [i0.to(device) for i0 in inputs]
            
            # loss, _, _, _, _,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers,inputs)
            # loss, _, _, _, _,_,_,_ = model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers)
            loss, _, _, _, ids,_,_,_ = model(**batch)
            # print(loss.device)
            # print(layer_outputs[0].device)
            # print(layer_outputs[0].device)
            
            
            # print(loss.device)
            # print(ids)
            all_loss = loss
            # print(layers_o[7].shape)
            if sparsity_rate and i%sparsity_rate == 0:
                
                f_c = 0
                for l in range(config.num_hidden_layers):
                    FN_loss = 0.0
                    for exp in range(config.num_experts):
                        # FNNP1 = sum([p.abs().mean() for p in model.layers[l].experts[exp].ffn.parameters()])
                        FNNP2 = sum([p.abs().mean() for p in model.layers[l].experts[exp].parameters()])
                        # FNNP = FNNP1+FNNP2
                        FNNP = FNNP2
                        FN_loss = FN_loss+ FNNP
                        all_loss = all_loss+1/(config.num_hidden_layers)*FN_loss
                        FN_Losses.append(accelerator.gather(FN_loss.repeat(config.batch_size)))
                        
                    
                # FN_loss = FN_loss/f_c
                # loss = loss + SPAR*FN_loss
            # print(FN_loss)
            if orth_rate and i%orth_rate == 0:
                
                for l in range(config.num_hidden_layers):
                    WI_loss = 0.0
                    w_c = 0
                    for exp1 in range(config.num_experts-1):
                        for exp2 in range(exp1+1,config.num_experts):
                            
                            for head in range(config.num_attention_heads):
                                Wi0 = model.layers[l].experts[exp1].attention.self.heads[head].weight
                                # print(Wi0[1].shape)
                                Wi0 = Wi0.view(-1)/Wi0.view(-1).norm()

                                Wj0 = model.layers[l].experts[exp2].attention.self.heads[head].weight
                                Wj0 = Wj0.view(-1)/Wj0.view(-1).norm()
                                # print(Wi0.shape)


                                WI_loss = WI_loss+(Wi0*Wj0).abs().sum()
                                # print("Wi0*Wj0",(Wi0*Wj0).sum())
                                w_c = w_c+1
                                # print(WI_loss)
                            # Wi1 = model.layers[l].experts[exp1].attention.self.WO.weight.view(-1)
                            # Wi1 = Wi1.view(-1)/Wi1.view(-1).norm()
                            # Wj1 = model.layers[l].experts[exp2].attention.self.WO.weight.view(-1)
                            # Wj1 = Wj1.view(-1)/Wj1.view(-1).norm()
                            
                            # WI_loss = WI_loss+(Wi1*Wj1).abs().sum()
                            # w_c = w_c+1
                    WI_loss = WI_loss/w_c
                    all_loss = all_loss+1/(config.num_hidden_layers)*WI_loss
                    WI_Losses.append(accelerator.gather(WI_loss.repeat(config.batch_size)))
                    # print(WI_loss)

            # IDS = get_routes_ids(ids,config)
            # for c in IDS:
            #     routes_history[c] += 1
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            optimizer.zero_grad()
            accelerator.backward(all_loss)
            optimizer.step()
            lr_scheduler.step()    

            if steps%100 == 0:
                print(f"steps:{steps}")
                print(ids)
                loss_train = torch.mean(torch.cat(losses)[:len(losses)])
                losses = []
                

                # loss_valid = validate(model, val_loader, accelerator, device, cluster_centers,model0,device2)
                # loss_valid = validate(model, val_loader, accelerator, device, cluster_centers)
                loss_valid = validate(model, val_loader, accelerator, device)
                # ahead_train = validate(model, ahead_val_loader, accelerator, device, cluster_centers,router)
                # routes_history = torch.tensor(routes_history)
                # listss = torch.argsort(routes_history,descending=True)[:10]
                # loss_test = validate(model, test_loader, accelerator)
                # accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Test Loss: {loss_test}')
                if sparsity_rate and orth_rate:
                    loss_L1 = torch.mean(torch.cat(FN_Losses)[:len(FN_Losses)])
                    loss_ORTH = torch.mean(torch.cat(WI_Losses)[:len(WI_Losses)])
                    
                    WI_Losses = []
                    FN_Losses = []
                    accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, L1 Loss: {loss_L1}, ORTH Loss:{loss_ORTH}')
                elif sparsity_rate and not orth_rate:
                    loss_L1 = torch.mean(torch.cat(FN_Losses)[:len(FN_Losses)])
                    # WI_Losses = []
                    FN_Losses = []
                    # loss_ORTH = torch.mean(torch.cat(WI_Losses)[:len(WI_Losses)])
                    accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, L1 Loss: {loss_L1}')
                elif not sparsity_rate and orth_rate:
                    # loss_L1 = torch.mean(torch.cat(FN_Losses)[:len(FN_Losses)])
                    loss_ORTH = torch.mean(torch.cat(WI_Losses)[:len(WI_Losses)])
                    WI_Losses = []
                    # FN_Losses = []
                    accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, ORTH Loss:{loss_ORTH}')
                else:
                    # loss_L1 = torch.mean(torch.cat(FN_Losses)[:len(FN_Losses)])
                    # loss_ORTH = torch.mean(torch.cat(WI_Losses)[:len(WI_Losses)])
                    accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}')
                if accelerator.is_local_main_process:
                    writer.add_scalar('perplexity_train_epoch', loss_train, steps)
                    writer.add_scalar('perplexity_valid', loss_valid, steps)
                    # writer.add_scalar('perplexity_ahead', ahead_train, epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], steps)
            if steps%100000 == 0 and steps>0:
                # torch.save(model,'0120_AngeL_rose_model_4expert_seq128_batchsize64_WIKI103_WP10_LAR3_%dsteps.pth'%steps)
                torch.save(model,'0128_AngeL_rose_model_2t_seq128_batchsize64_LAR_autoinit_WP128000_%dsteps.pth'%steps)

    # accelerator.save_state('./output-formal-1027-new_model-stage1-freeze_embed')
    # torch.save(model,'0120_AngeL_rose_model_4expert_seq128_batchsize64_WIKI103_WP10_LAR3.pth')
    torch.save(model,'0128_AngeL_rose_model_2t_seq128_batchsize64_LAR_autoinit_WP128000.pth')
    

if __name__ == "__main__":
    set_seed(42)
    
    config = BertConfig.from_json_file('config/AngeL_rose.json')
    # config = BertConfig.from_json_file('config/AngeL_rose2.json')

    # dataset = RestaurantForLM_small(config=config)
    # dataset = MoMoE_MIXED_WIKI103_0124(config = config)
    # ahead_dataset = MoMoE_MIXED_WIKI103_0124(config = config)
    dataset = MoMoE_MIXED_0128(config = config)
    ahead_dataset = MoMoE_MIXED_0128(config = config)
    torch.cuda.set_device(6)
    device = torch.device("cuda")
    # model = base_models.AngeL_rose_model_tokens_cluster(config=config)
    # model = base_models.AngeL_rose_model(config=config)
    # model = base_models.AngeL_rose_model_0126(config=config)
    # model = base_models.AngeL_rose_model_prenorm(config=config)
    model = base_models.AngeL_rose_model_salary1062(config=config)




    # router = base_models.BertWithSavers(config=config)
    # router.to(device)
    model.to(device)
    # model = nn.DataParallel(model)

    
    train(model=model, num_epochs=3, dataset=dataset, device=device, ahead_dataset = ahead_dataset)