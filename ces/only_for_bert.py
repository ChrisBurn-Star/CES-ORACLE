import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import MoMoE_rebuttal,Wikipedia_small,PubMedForLM,LegalForLM,Wikipedia,MoMoE_longtailed,MoMoE_MIXED_LEGAL_REVIEW,MoMoE_LARGE_SPECIFIC_GENERAL,Wikitxt103ForLM_80W,MoMoE_FEWER_SPECIFIC_GENERAL,MoMoE_MIXED_LEGAL_PUBMED,MoMoE_MIXED_0128,MoMoE_MIXED_WIKI103_0124,MoMoE_MIXED,MoMoE_MIXED_WARMUP,MoMoE_WIKI103,MoMoE_WIKI103_WARMUP,MixedData_1211,MixedData_1211_0,RestaurantforLM_1103, MixedData, MixedData_stage1, ACLForLM,old_MixedData_after_stage1, Mixdata_1103, Wikitext,ACLForLM_1103,Mixdata_1115,Review_1103,Wikitxt103ForLM_1103,Wikitxt103ForLM_0102_warmup,Wikitxt103ForLM_0102_bert,MixedData_0110_0,MixedData_0110_1
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim
from sklearn.decomposition import PCA
import torch
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import util
from accelerate import DistributedDataParallelKwargs as DDPK
from accelerate import load_checkpoint_and_dispatch
import seaborn as sns
import os


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
        # batch = {key: tensor.to(device) for key, tensor in batch.items()}      
        with torch.no_grad():
            loss = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity


def get_gradient_norms(model):
    """Utility function to get gradient norms of a model."""
    return [param.grad.norm().item() for param in model.parameters() if param.grad is not None]


def train(model, num_epochs, dataset, device,LEGAL):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    # model = torch.load('1127-bert-only.pth')
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    # val_loader2 = ahead1.val_loader
    # val_loader1 = dataset.val_loader_domain2
    # val_loader2 = dataset.val_loader_domain2
    # val_loader3 = dataset.val_loader_domain_common
    # val_loader_domain1_ntk = dataset.val_loader_domain1_ntk
    # val_loader_domain2_ntk = dataset.val_loader_domain2_ntk
    # val_loader3 = ahead2.val_loader
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    # accelerator = Accelerator(kwargs_handlers=[DDPK(find_unused_parameters=True)])

    # writer = SummaryWriter('tensorboard_0223/0226-bert-batch64_seq128-MIXED_LEGAL_PUBMED-pretrain-lr2e-4')
    writer = SummaryWriter('/home/jxzhou/PLM_PER/iclr2025_codes/tensorboard/0918-bert-png')
    # load_checkpoint_and_dispatch(model, '/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0401/0418-BERT-768ffns-wikipedia-1.5e-4',device_map="auto")
    # model0 = base_models.BertForMLM(config)
    # load_checkpoint_and_dispatch(model0, '/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0429/0429-BERT-768ffns-wikipedia-1.5e-4',device_map={"":device})
    # model.load_state_dict(model0.state_dict())
    # model0 = base_models.BertForMLM(config=config)
    # load_checkpoint_and_dispatch(model0, '/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_REBUTALL/bert-copyparams',device_map={"":device})
    # model.bert.load_state_dict(model0.bert.state_dict())


    # checkpoint = torch.load(os.path.join('/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0514/BERT-768ffns-mixed-small-1.5e-4', 'pytorch_model.bin'))
    # model.load_state_dict(checkpoint)
    # model = torch.load('0119-bert-WIKI103-ALL.pth')
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    model, optimizer, lr_scheduler, train_loader, val_loader= accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    # model, optimizer, lr_scheduler, train_loader, val_loader,val_loader1,val_loader2= accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader,val_loader1,val_loader2)
    # model, optimizer, lr_scheduler, train_loader, val_loader,val_loader1,val_loader2,val_loader3,val_loader_domain1_ntk,val_loader_domain2_ntk = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader,val_loader1,val_loader2,val_loader3,val_loader_domain1_ntk,val_loader_domain2_ntk)
    
    # model.to(device)
    steps = 0
    # accelerator.save_state('MODELS_0502/BERT-768ffns-wikipedia-large-1.5e-4-checkpoints%d'%steps)
    losses = []
    t = 0

    # for i, batch in enumerate(val_loader_domain1_ntk):
    
    #     batchl = {k: v[:1, :] for k, v in batch.items()}
    #     break
    print(len(train_loader))

    for epoch in range(num_epochs):
        model.train()
        
        """train origin bert (MLM only)"""
        
        for i, batch in enumerate(train_loader):
            steps += 1
            # print(i)
            # if steps == 5001:
            # #     steps =0
            #     break
            # print(batch['attention_mask'].shape)
            
            # batch = {key: tensor.to(device) for key, tensor in batch.items()}
            # print(next(model.parameters()).device)
            # for key, tensor in batch.items():
            #     print(f"{key} is on {tensor.device}")
            loss= model(**batch)
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()    
            if steps%100 == 0:
                loss_train = torch.mean(torch.cat(losses)[:len(losses)])
                losses = []
                loss_valid = validate(model, val_loader, accelerator, device)
                # loss_valid1 = validate(model, val_loader1, accelerator, device)
                
                # loss_valid2 = validate(model, val_loader2, accelerator, device)
                # loss_valid3 = validate(model, val_loader3, accelerator, device)


                accelerator.print(f'Epoch:{epoch} ({i} Updates), steps:{steps}, Train Loss: {loss_train}, Valid Loss: {loss_valid}')

                if accelerator.is_local_main_process:
                    writer.add_scalar('perplexity_train_epoch', loss_train, steps)
                    writer.add_scalar('perplexity_valid', loss_valid, steps)
                    # writer.add_scalar('leagl_perplexity_valid', loss_valid1, steps)

                    # writer.add_scalar('pubmed_perplexity_valid', loss_valid2, steps)
                    # writer.add_scalar('self_perplexity_valid', loss_valid1, steps)


                    writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], steps)
                torch.save(model,f'/home/jxzhou/PLM_PER/iclr2025_codes/scripts/checkpoint/BERT-baseline-2Mdata/steps_{steps}.pth')
            # if steps%7000 == 0:
            #     # loss_valid_f = validate(model, val_loader_f, accelerator, device)
            #     # if accelerator.is_local_main_process:
            #     #     writer.add_scalar('more_perplexity_valid', loss_valid_f, steps)
            #     accelerator.save_state('MODELS_0514/BERT-768ffns-mixed-small-1.5e-4-checkpoints%d'%steps)
            # if steps%100 == 0:
            #     accelerator.save_state('/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_REBUTALL/BERT_MED_0801-2-checkpoints%d'%steps)


            #####NTKS
            # if steps%2000 == 0 and steps>1000:
            #     _, _, _,t = get_ntk_by_layer(model,val_loader_domain1_ntk,steps,"legal",batchl,t)
            #     _, _, _,t = get_ntk_by_layer(model,val_loader_domain2_ntk,steps,"bio",batchl,t)

    # accelerator.save_state('/home/jxzhou/PLM_PER/iclr2025_codes/scripts//checkpoint/BERT-baseline-2Mdata/')
    torch.save(model,f'/home/jxzhou/PLM_PER/iclr2025_codes/scripts/checkpoint/BERT-baseline-2Mdata/steps_{steps}.pth')

    # torch.save(model,'MODELS_0401/0401-BERT-3072ffns-wikipedia-1.5e-4.pth')




def pca_projection_with_torch(input,projects,k0):
    input = input.cpu().numpy()
    matrix = projects.cpu().numpy()

    pca = PCA(n_components=k0,whiten=True)  # 假设你想要降维到2维
    pca.fit(matrix)

    # 假设你有一个单独的张量
    if not np.allclose(input, np.zeros_like(input)):
        tensor = input / np.linalg.norm(input)
    else:
        tensor = input

    # 使用PCA的转换基来转换张量
    transformed_tensor = torch.tensor(pca.transform(tensor))
    norm_t = torch.sum(transformed_tensor)/k0
    return transformed_tensor, norm_t

def pca_similarity_with_torch(input,projects,k0):
    input = input.cpu().numpy()
    matrix = projects.cpu().numpy()

    pca = PCA(n_components=k0,whiten=True)  # 假设你想要降维到2维
    pca.fit(matrix)
    transformed_basis = pca.components_

    # 假设你有一个单独的张量
    if not np.allclose(input, np.zeros_like(input)):
        tensor = input / np.linalg.norm(input)
    else:
        tensor = input
    # 使用PCA的转换基来转换张量
    transformed_tensor = torch.tensor(pca.transform(tensor))

    cosine_similarities = cosine_similarity(tensor, transformed_basis)

    cosine_similarities = torch.tensor(np.mean(cosine_similarities))
    
    return transformed_tensor, cosine_similarities




def get_ntk_by_layer(model,loader,steps,doamin,longtailed_data,t):

    L , H = 12,8
    NTKs = [[[] for k in range(H)]for j in range(L)]
    longtailed_NTKs = [[[] for k in range(H)]for j in range(L)]
    J_GRADS = [[[] for k in range(H)]for j in range(L)]

    NTK = torch.zeros(L,H)
    longtailed_NTK = torch.zeros(L,H)
    longtailed_sim = torch.zeros(L,H)

    params_that_need_grad = []
    for param in model.parameters():
        if param.requires_grad:
            params_that_need_grad.append(param.requires_grad)
            param.requires_grad = False # first set all gradients to not calculate, time saver
        else:
            params_that_need_grad.append(param.requires_grad)
    
    for index, (name, param) in enumerate(model.named_parameters()):
        if not params_that_need_grad[index]: #if it didnt need a grad, we can skip it.
            continue
        
        if 'heads' in name and 'attention' in name:
            param.requires_grad = True #we only care about this tensors gradients in the loop
            this_grad = []
            longtailed_data_this_grad = []
            splited = name.split('.')
            l = int(splited[splited.index('layers') + 1])
            # t = int(splited[splited.index('attentions') + 1])
            h = int(splited[splited.index('heads') + 1])
            print(l,h)
            if l >= L:break
            print(name)

            for i, batch in enumerate(loader):
                if i >= 1: break
                


                _, logits = model(**batch)

                for i in range(len(logits)):
                    # print(i)
                    logits[i:i+1].backward(torch.ones_like(logits[i:i+1]), create_graph=True)
                    this_grad.append(param.grad.detach().reshape(-1).clone())
                    model.zero_grad()
                    torch.cuda.empty_cache()

                
                
                _, logits= model(**longtailed_data)

                for i in range(len(logits)):
                    # print(i)
                    logits[i:i+1].backward(torch.ones_like(logits[i:i+1]), create_graph=True)
                    longtailed_data_this_grad.append(param.grad.detach().reshape(-1).clone())
                    model.zero_grad()
                    torch.cuda.empty_cache()

            J_layer = torch.stack(this_grad) # [N x P matrix] #this will go against our notation, but I'm not adding
            longtailed_J_layer = torch.stack(longtailed_data_this_grad) # [N x P matrix] #this will go against our notation, but I'm not adding
 
            NTKs[l][h].append(J_layer @ J_layer.T )# An extra transpose operation to my code for us to feel better
            J_GRADS[l][h].append(J_layer)
            # print(NTKs[l][t][h])
            longtailed_NTKs[l][h].append(longtailed_J_layer)
            param.requires_grad = False
            model.zero_grad()
            torch.cuda.empty_cache()

            

     
    #reset the model object to be how we started this function
    for i, param in enumerate(model.parameters()):
        if params_that_need_grad[i]:
            param.requires_grad = True
    
    for l in range(L):

            for h in range(H):
                # if len(NTKs[l][w][h]) == 0:
                #     NTK[l][w][h] = torch.tensor(0)
                
                # print(util.lmax(NTKs[l][w][h][0]))
                NTK[l][h] = util.lmax(NTKs[l][h][0])
                # print(longtailed_NTKs[l][t][h][0])
                _,longtailed_NTK[l][h] = pca_projection_with_torch(longtailed_NTKs[l][h][0],J_GRADS[l][h][0],1)
                # print(longtailed_NTK[l][w][h])
                _,longtailed_sim[l][h] = pca_similarity_with_torch(longtailed_NTKs[l][h][0],J_GRADS[l][h][0],1)
                # print(longtailed_sim[l][w][h])
            # print(NTK[l][w])
            # print(longtailed_NTK[l][w])
            # print(longtailed_sim[l][w])


    lambdamax_layers = NTK

    plt.figure(t,dpi=120)
    sns.heatmap(data=lambdamax_layers[:,:],
                vmin=1e11,
                vmax=1e16,
                cmap=plt.get_cmap('Blues')
        )
    plt.title('LAMBDAmax_IN_HEADS_EVERY_LAYERS') 
    plt.savefig('NTKS-0330-BERT/NTK-lambdamax-%dsteps-%s.png'%(steps,doamin)) 
    t+=1

    plt.figure(t,dpi=120)
    sns.heatmap(data=longtailed_NTK[:,:],
                vmin=-1,
                vmax=1,
                cmap=plt.get_cmap('Blues')
        )
    plt.title('PROJECTIONS_IN_HEADS_EVERY_LAYERS') 
    plt.savefig('NTKS-0330-BERT/NTK-projection-%dsteps-%s.png'%(steps,doamin)) 
    t+=1
    plt.figure(t,dpi=120)
    sns.heatmap(data=longtailed_sim[:,:],
                vmin=-1,
                vmax=1,
                cmap=plt.get_cmap('Blues')
        )
    plt.title('SIMILARITY_IN_HEADS_EVERY_LAYERS') 
    plt.savefig('NTKS-0330-BERT/NTK-similarity-%dsteps-%s.png'%(steps,doamin)) 
    t+=1
    print(steps,"finished")
    return NTKs, NTK,lambdamax_layers,t



def main():
    set_seed(45)
    
    config = BertConfig.from_json_file('config/bert.json')
    # dataset = RestaurantForLM_small(config=config)
    dataset = Wikipedia(config = config)
    ahead1 = Wikipedia(config)
    # torch.cuda.set_device(5)
    # ahead2 = Mixdata_1103(config)
    # ahead_dataset = old_MixedData_after_stage1(config = config)
    device = torch.device("cuda")
    model = base_models.BertForMLM(config=config)
    # model = base_models.BertForMLM_prenorm(config=config)

    # model.to(device)
    # model = nn.DataParallel(model)
    
    train(model=model, num_epochs=1, dataset=dataset, device=device,ahead1=ahead1)

def count_parameters_detailed(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params
if __name__ == "__main__":
    # main()
    set_seed(28)
    
    config = BertConfig.from_json_file('config/bert.json')
    # dataset = PubMedForLM(config = config)
    # ahead1 = Wikipedia(config)
    
    # ahead2 = Mixdata_1103(config)
    # ahead_dataset = old_MixedData_after_stage1(config = config)
    LEGAL = 100000
    # dataset = Wikipedia_small(config=config,KKK = LEGAL)

    dataset = MoMoE_MIXED_LEGAL_PUBMED(config = config)

    # # ahead_dataset = Wikipedia_Wids(config = config)
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")
    model = base_models.BertForMLM(config=config)
    # model = base_models.BertForMLM_prenorm(config=config)

    # model.to(device)
    # model = nn.DataParallel(model)
    
    train(model=model, num_epochs=1, dataset=dataset, device=device,LEGAL = LEGAL)

    # total_params, trainable_params, non_trainable_params = count_parameters_detailed(model)
    # print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")
    # print(f"Non-trainable parameters: {non_trainable_params}")