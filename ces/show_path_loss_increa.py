import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import ACLForLM_1103,Mixdata_1115
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim
import matplotlib.pyplot as plt
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
def get_routes_ids(expert_ids,config):
    IDS = []
    for i in range(expert_ids[0].shape[0]):
        c = 0
        for j in range(len(expert_ids)):
            c+=expert_ids[j][i]*2**j
        IDS.append(c)
    return IDS

def get_the_other_routes_ids(expert_ids,config):
    IDS = []
    for i in range(expert_ids[0].shape[0]):
        c = 0
        for j in range(len(expert_ids)):
            if expert_ids[j][i] == 1:
                t = 0
            else:
                t = 1
            c+=t*2**j
        IDS.append(c)
    return IDS

def get_other_routes_ids(expert_ids,config):
    IDS = []
    for i in range(expert_ids[0].shape[0]):
        c = 0
        for j in range(len(expert_ids)):
            if j <6:
                t = expert_ids[j][i]
            else:

                if expert_ids[j][i] == 1:
                    t = 0
                else:
                    t = 1
            c+=t*2**j
        IDS.append(c)
    return IDS


def train(model, num_epochs, dataset, device, ahead_dataset):
    path_replay = 1
    other_routes = 0
    REPLAY_STEPS = 20
    M = [560,  561, 2597]
    # [ 564,  629,565, 2661,  549,  544,  560,  561, 2597, 609,820,821]
    model = torch.load('1115_vermilion_model_satge1.pth')
    cluster_centers = load_layer_data('1115-layer_centers-2expert.pth')
    test = load_layer_data('1115-replay_data_vermilion_models-2experts.pth')

    print(len(cluster_centers), cluster_centers[9].shape)



    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    ahead_train_loader, ahead_val_loader = ahead_dataset.train_loader, ahead_dataset.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    writer = SummaryWriter('1115_vermilion_model_satge2-test')

    
    num_updates = num_epochs * len(train_loader)
    # lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    model.to(device)

    for i, batch in enumerate(ahead_train_loader):
        if i == 22:
            batch0 = batch.to('cuda')
    for epoch in range(num_epochs):
        model.train()
        # losses = []
        routes_history = [0 for i in range(2**12)]
        for i, batch in enumerate(train_loader):
            
            # batch = {'input_ids': torch.cat((batch['input_ids'],batch0['input_ids']),dim = 0), 'attention_mask': torch.cat((batch['attention_mask'],batch0['attention_mask']),dim = 0), 'labels': torch.cat((batch['labels'],batch0['labels']),dim = 0)}
            # batch = batch0
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            

            loss, _, _ ,_, EXPERT_IDS= model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers)
            # _, _, _ ,_, EXPERT_IDS= model(batch['input_ids'],batch['attention_mask'], batch['labels'], cluster_centers)
            
            
            # losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            # print(EXPERT_IDS)


            IDS = get_routes_ids(EXPERT_IDS,config)
            other_IDS = get_the_other_routes_ids(EXPERT_IDS,config)

            if (1468 in IDS) or (1340 in IDS):
                print(epoch, i)
            # if (1334 in IDS) or (1340 in IDS) or (1342 in IDS) or (1462 in IDS) or (1468 in IDS) or (1470 in IDS) or (2625 in IDS) or (2627 in IDS) or (2633 in IDS) or (2753 in IDS) or (2755 in IDS) or (2561 in IDS):
            #     print('####################')
            #     print(i)
            #     break
                PP = []
                PP.append(str(IDS[0].item()))
                PP.append(str(4095-IDS[0].item()))
                PP = PP+[str(m) for m in M]
                # print(IDS)
                OLOSS1 = 0
                MS0 = []
                MS1 = []
                MS = []
                for c in IDS:
                    for l in range(len(test[c])):
                        same_loss1, _, _ ,_, _= model(test[c][l]['input_ids'].view(1,config.seq_len),test[c][l]['attention_mask'].view(1,config.seq_len), test[c][l]['labels'].view(1,config.seq_len), cluster_centers)
                        OLOSS1 = OLOSS1 + same_loss1
                OLOSS1 = OLOSS1/len(test[c])

                OLOSS2 = 0
                for k in other_IDS:
                    for l in range(len(test[k])):
                        opposite_loss1, _, _ ,_, _= model(test[k][0]['input_ids'].view(1,config.seq_len),test[k][0]['attention_mask'].view(1,config.seq_len), test[k][0]['labels'].view(1,config.seq_len), cluster_centers)
                        OLOSS2 = OLOSS2 + opposite_loss1
                OLOSS2 = OLOSS2/len(test[k])
                dis = []
                for lm in range(len(M)):
                    m = M[lm]
                    print(lm)
                    OLOSS3 = 0
                    for k in other_IDS:
                        for l in range(len(test[m])):
                            opposite2_loss1, _, _ ,_, _= model(test[m][0]['input_ids'].view(1,config.seq_len),test[m][0]['attention_mask'].view(1,config.seq_len), test[m][0]['labels'].view(1,config.seq_len), cluster_centers)
                            OLOSS3 = OLOSS3 + opposite2_loss1
                    OLOSS3 = OLOSS3/len(test[m])
                    dis.append(OLOSS3)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()

                PLOSS1 = 0
                for c in IDS:
                    for l in range(len(test[c])):
                        same_loss1, _, _ ,_, _= model(test[c][l]['input_ids'].view(1,config.seq_len),test[c][l]['attention_mask'].view(1,config.seq_len), test[c][l]['labels'].view(1,config.seq_len), cluster_centers)
                        PLOSS1 = PLOSS1 + same_loss1
                PLOSS1 = PLOSS1/len(test[c])
                
                PLOSS2 = 0
                for k in other_IDS:
                    for l in range(len(test[k])):
                        opposite_loss1, _, _ ,_, _= model(test[k][0]['input_ids'].view(1,config.seq_len),test[k][0]['attention_mask'].view(1,config.seq_len), test[k][0]['labels'].view(1,config.seq_len), cluster_centers)
                        PLOSS2 = PLOSS2 + opposite_loss1
                PLOSS2 = PLOSS2/len(test[k])
                det_loss0 = PLOSS1 - OLOSS1
                det_loss1 = PLOSS2 - OLOSS2
                MS0.append(det_loss0.item())
                MS0.append(det_loss1.item())
                for lm in range(len(M)):
                    m = M[lm]
                    PLOSS3 = 0
                    for k in other_IDS:
                        for l in range(len(test[m])):
                            opposite2_loss1, _, _ ,_, _= model(test[m][0]['input_ids'].view(1,config.seq_len),test[m][0]['attention_mask'].view(1,config.seq_len), test[m][0]['labels'].view(1,config.seq_len), cluster_centers)
                            PLOSS3 = PLOSS3 + opposite2_loss1
                    PLOSS3 = PLOSS3/len(test[m])
                    det_loss2 = PLOSS3 - dis[lm]
                    MS0.append(det_loss2.item())
                
                del opposite2_loss1
                
                SAME_LOSS = 0
                for c in IDS:
                    for l in range(len(test[c])):
                        same_loss2, _, _ ,_, _= model(test[c][l]['input_ids'].view(1,config.seq_len),test[c][l]['attention_mask'].view(1,config.seq_len), test[c][l]['labels'].view(1,config.seq_len), cluster_centers)
                        SAME_LOSS = SAME_LOSS + same_loss2
                SAME_LOSS = SAME_LOSS/len(test[c])
            
                OPPO_LOSS = 0
                for k in other_IDS:
                    for l in range(len(test[k])):

                        opposite_loss2, _, _ ,_, _= model(test[k][0]['input_ids'].view(1,config.seq_len),test[k][0]['attention_mask'].view(1,config.seq_len), test[k][0]['labels'].view(1,config.seq_len), cluster_centers)
                        OPPO_LOSS = OPPO_LOSS + opposite_loss2
                OPPO_LOSS = OPPO_LOSS/len(test[k])
                optimizer.zero_grad()
                # OPPO_LOSS.backward()
                SAME_LOSS.backward()
                optimizer.step()

                
                KLOSS1 = 0
                for c in IDS:
                    for l in range(len(test[c])):
                        same_loss1, _, _ ,_, _= model(test[c][l]['input_ids'].view(1,config.seq_len),test[c][l]['attention_mask'].view(1,config.seq_len), test[c][l]['labels'].view(1,config.seq_len), cluster_centers)
                        KLOSS1 = KLOSS1 + same_loss1
                KLOSS1 = KLOSS1/len(test[c])

                KLOSS2 = 0
                for k in other_IDS:
                    for l in range(len(test[k])):
                        opposite_loss1, _, _ ,_, _= model(test[k][0]['input_ids'].view(1,config.seq_len),test[k][0]['attention_mask'].view(1,config.seq_len), test[k][0]['labels'].view(1,config.seq_len), cluster_centers)
                        KLOSS2 = KLOSS2 + opposite_loss1
                KLOSS2 = KLOSS2/len(test[k])
                det_loss3 = KLOSS1 - OLOSS1
                det_loss4 = KLOSS2 - OLOSS2
                MS1.append(det_loss3.item())
                MS1.append(det_loss4.item())
                del opposite_loss1
                for lm in range(len(M)):
                    m = M[lm]
                    KLOSS3 = 0
                    for k in other_IDS:
                        for l in range(len(test[m])):
                            opposite2_loss1, _, _ ,_, _= model(test[m][0]['input_ids'].view(1,config.seq_len),test[m][0]['attention_mask'].view(1,config.seq_len), test[m][0]['labels'].view(1,config.seq_len), cluster_centers)
                            KLOSS3 = KLOSS3 + opposite2_loss1
                    KLOSS3 = KLOSS3/len(test[m])
                    det_loss5 = KLOSS3 - dis[lm]
                    MS1.append(det_loss5.item())


                # print(MS0[3].type())
                print(PP)
                print(MS0)
                print(MS1)
                plt.figure(i)
                plt.bar(PP, MS0,color='skyblue',width=0.5)
                # plt.scatter(PP, MS0)
                plt.title('without replay')
                plt.xlabel('path ids')
                plt.ylabel('loss_difference')
                plt.savefig('1117-without-replay-i%d-3'%i + '.png')
                plt.show()
                plt.figure(i+10000)
                plt.bar(PP, MS1,color='skyblue',width=0.5)
                # plt.scatter(PP, MS1)
                plt.title('with replay nearest path')
                plt.xlabel('path ids')
                plt.ylabel('loss_difference')
                plt.savefig('1117-with-replay-nearest-path-i%d-3'%i + '.png')
                plt.show()
                # print(f'path:{c} withOUT reply forgetting:(nearest:{det_loss0}, farthest:{det_loss1}), another:{det_loss2} ; with reply nearest and farthest forgetting:(nearest:{det_loss3}, farthest:{det_loss4}), another:{det_loss5} ')
                print(f'path:{c} withOUT reply forgetting:(nearest:{det_loss0}, farthest:{det_loss1}), another:{det_loss2} ;with reply nearest forgetting:(nearest:{det_loss3}, farthest:{det_loss4}), another:{det_loss5}')
                break

        
    # torch.save(model,'1115_vermilion_model_satge2.pth')
    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/path_check.json')
    dataset = ACLForLM_1103(config = config)
    ahead_dataset = Mixdata_1115(config = config)
    device = torch.device("cuda")
    model = base_models.vermilion_model(config=config)
    model.to(device)

    train(model=model, num_epochs=1, dataset=dataset, device=device, ahead_dataset = ahead_dataset)