import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import MoMoE_MIXED_LEG,MoMoE_MIXED_REV,MoMoE_MIXED_ACL,MoMoE_MIXED_RES,MoMoE_MIXED_WIKI103_0124,MoMoE_MIXED,MoMoE_WIKI103,MixedData_1211,MixedData_0110_1,MixedData_1211_1,RestaurantForLM_small, MixedData,MixedData_stage1,Mixdata_1103,Mixdata_1115,Wikitxt103ForLM_1103,Wikitxt103ForLM_0102_rose
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



def validate(model, val_loader, accelerator, device):
    losses = []

    for i, batch in enumerate(val_loader):  

        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        with torch.no_grad():
            
            loss, _, _, _, _,_,_,_ = model(**batch)
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

def train(model, num_epochs, dataset, device, dataset2):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    
    freze_emb = 0 
    AngeL = 0
    sparsity_rate = 0
    SPAR = 1
    orth_rate = 0
    ORTH = 1
    same_in_param = 0

    with_bert_param = 1
    # model0 = base_models.BertForMLM_toshow(config=config)
    # model01 = torch.load('0119-bert-MIXED-WP3.pth')
    # model0.load_state_dict(model01.state_dict())
    # device2 = torch.device("cuda:3")
    # model0.to(device2)

    # model0 = torch.load('0119-bert-WIKI103-WP10.pth')
    # cluster_centers = load_layer_data('0119-layer_centers-2t-MIXED-WP3.pth')

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
    train_loader2, val_loader2 = dataset2.train_loader, dataset2.val_loader
    # ahead_train_loader = ahead_dataset.train_loader
    # ahead_val_loader = ahead_dataset.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    # writer = SummaryWriter('tensorboard_0119/0120_AngeL_rose_model_4expert_seq128_batchsize64_WIKI103_WP10_LAR3')
    writer = SummaryWriter('tensorboard_0119/0128_AngeL_rose_model_2t_seq128_batchsize64_LAR_correctdistribution')


    
    num_updates = num_epochs * len(train_loader)*2
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    # model, optimizer, lr_scheduler, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, test_loader)
    model, optimizer, lr_scheduler, train_loader, val_loader,train_loader2, val_loader2= accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader,train_loader2, val_loader2)
    model.to(device)
    # device2 = torch.device("cuda:3")
    # model0.to(device2)
    # router.to(device)
    # if freze_emb:
    #     for para in model.embeddings.parameters():
    #         para.requires_grad = False
    steps = 0
    losses = []
    losses2 = []
    WI_Losses = []
    FN_Losses = []
    for epoch in range(num_epochs):
        
        model.train()
        
        
        # routes_history = [0 for i in range(4**12)]
        for i, (batch1,batch2) in enumerate(zip(train_loader,train_loader2)):
            

            batch1 = {key: tensor.to(device) for key, tensor in batch1.items()}
            batch2 = {key: tensor.to(device) for key, tensor in batch2.items()}
            # print(epoch, i)

            
            steps+= 1
            model.route = 0
            loss1, _, _, _, ids,_,_,_ = model(**batch1)
            losses.append(accelerator.gather(loss1.repeat(config.batch_size)))
            optimizer.zero_grad()
            accelerator.backward(loss1)
            optimizer.step()
            lr_scheduler.step()  

            steps+= 1
            model.route = 1
            loss2, _, _, _, ids,_,_,_ = model(**batch2)  
            losses2.append(accelerator.gather(loss2.repeat(config.batch_size)))
            optimizer.zero_grad()
            accelerator.backward(loss2)
            optimizer.step()
            lr_scheduler.step() 

            if steps%100 == 0:
                # print(f"steps:{steps}")
                # print(ids)
                loss_train = torch.mean(torch.cat(losses)[:len(losses)])
                loss_train2 = torch.mean(torch.cat(losses2)[:len(losses2)])
                losses = []
                losses2 = []
                

                # loss_valid = validate(model, val_loader, accelerator, device, cluster_centers,model0,device2)
                # loss_valid = validate(model, val_loader, accelerator, device, cluster_centers)
                
                model.route = 0
                loss_valid = validate(model, val_loader, accelerator, device)
                model.route = 1
                loss_valid2 = validate(model, val_loader2, accelerator, device)
                # ahead_train = validate(model, ahead_val_loader, accelerator, device, cluster_centers,router)
                # routes_history = torch.tensor(routes_history)
                # listss = torch.argsort(routes_history,descending=True)[:10]
                # loss_test = validate(model, test_loader, accelerator)
                # accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Test Loss: {loss_test}')
                loss_all_valid = (loss_valid+loss_valid2)/2
                accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss1: {loss_train}, Valid Loss1: {loss_valid}, Train Loss2: {loss_train2}, Valid Loss2: {loss_valid2}')
                if accelerator.is_local_main_process:
                    writer.add_scalar('perplexity_train_epoch1', loss_train, steps)
                    writer.add_scalar('perplexity_train_epoch2', loss_train2, steps)
                    writer.add_scalar('perplexity_valid1', loss_valid, steps)
                    writer.add_scalar('perplexity_valid2', loss_valid2, steps)
                    writer.add_scalar('perplexity_valid3', loss_all_valid, steps)
                    
                    # writer.add_scalar('perplexity_ahead', ahead_train, epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], steps)
            if steps%100000 == 0 and steps>0:
                # torch.save(model,'0120_AngeL_rose_model_4expert_seq128_batchsize64_WIKI103_WP10_LAR3_%dsteps.pth'%steps)
                torch.save(model,'0128_AngeL_rose_model_2t_seq128_batchsize64_LAR_correctdistribution_%dsteps.pth'%steps)

    # accelerator.save_state('./output-formal-1027-new_model-stage1-freeze_embed')
    # torch.save(model,'0120_AngeL_rose_model_4expert_seq128_batchsize64_WIKI103_WP10_LAR3.pth')
    torch.save(model,'0128_AngeL_rose_model_2t_seq128_batchsize64_LAR_correctdistribution.pth')
    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/AngeL_rose.json')
    # config = BertConfig.from_json_file('config/AngeL_rose2.json')

    # dataset = RestaurantForLM_small(config=config)
    # dataset = MoMoE_MIXED_WIKI103_0124(config = config)
    # ahead_dataset = MoMoE_MIXED_WIKI103_0124(config = config)
    dataset = MoMoE_MIXED_LEG(config = config)
    dataset2 = MoMoE_MIXED_REV(config = config)
    torch.cuda.set_device(3)
    device = torch.device("cuda")
    # model = base_models.AngeL_rose_model_tokens_cluster(config=config)
    # model = base_models.AngeL_rose_model(config=config)
    model = base_models.AngeL_rose_model_0128(config=config)
    # model = base_models.AngeL_rose_model_prenorm(config=config)
    # model = base_models.AngeL_rose_model_salary1062(config=config)




    # router = base_models.BertWithSavers(config=config)
    # router.to(device)
    model.to(device)
    # model = nn.DataParallel(model)

    
    train(model=model, num_epochs=3, dataset=dataset, device=device, dataset2 = dataset2)