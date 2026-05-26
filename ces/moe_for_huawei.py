import torch
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.optim as optim
from transformer.Transformer_MOE import BertModel,BertModel_prenorm,BertModel_every2layers
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import BertConfig, get_cosine_schedule_with_warmup
from accelerate import Accelerator,load_checkpoint_and_dispatch
from accelerate import DistributedDataParallelKwargs as DDPK
from Dataset import Wikitext
from Dataset_new import MoE_huawei,Wikipedia_small,LegalForLM,PubMedForLM,Wikipedia,MoMoE_longtailed,MoMoE_MIXED_LEGAL_REVIEW,MoMoE_LARGE_SPECIFIC_GENERAL,MoMoE_MIXED_LEGAL_PUBMED,MoMoE_MIXED_0128,MoMoE_MIXED_REV,MoMoE_MIXED_LEG,MoMoE_MIXED_WIKI103_0124,MoMoE_MIXED,MoMoE_WIKI103,MixedData_1211,RestaurantforLM_1103,Review_1103,ACLForLM_1103,Mixdata_1103,Wikitxt103ForLM_1103,MoMoE_SINGLE,MoMoE_FEWER_SPECIFIC_GENERAL
from einops import rearrange
import util
import random
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.profiling.flops_profiler import FlopsProfiler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import os
class BertForMLM(nn.Module):
    def __init__(self, config):
        super(BertForMLM, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask, labels):
        output,ffn_sself,O,_,ROUTES = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores,ffn_sself,O,ROUTES
    


class BertForMLM_every2layers(nn.Module):
    def __init__(self, config):
        super(BertForMLM_every2layers, self).__init__()
        self.config = config
        self.bert = BertModel_every2layers(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask, labels):
        output,att_sself,O,_,_,_ = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores,att_sself,O


    
class BertForMLM_prenorm(nn.Module):
    def __init__(self, config):
        super(BertForMLM_prenorm, self).__init__()
        self.config = config
        self.bert = BertModel_prenorm(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask, labels):
        output,att_sself,O = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores,att_sself,O

# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling.ipynb
def train(model: BertForMLM, dataset, epoch):
    # accelerator: https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
    # accelerator = Accelerator(kwargs_handlers=[DDPK(find_unused_parameters=True)])
    accelerator = Accelerator()
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
    val_loader1 = dataset.val_loader_domain1
    val_loader2 = dataset.val_loader_domain2
    
    # load_checkpoint_and_dispatch(model, '/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0401/0417-MOE-768ffns-wikipedia-2e-4',device_map={"":device})

    # val_loader3 = dataset.val_loader_domain_common
    # val_loader_domain1_ntk = dataset.val_loader_domain1_ntk
    # val_loader_domain2_ntk = dataset.val_loader_domain2_ntk
    # test_loader = dataset.test_loader

    # config = model.config
    # model = torch.load('1221-moe-form-yubin-mix.pth')
    
    if accelerator.is_local_main_process:
        writer = SummaryWriter('/home/jxzhou/PLM_PER/BERT-CL-main/MOE_HUAWEI/MOE')

    num_epochs = epoch
    num_updates = num_epochs * len(train_loader)

    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_updates * 0.06,
        num_training_steps=num_updates,
    )
    # model = torch.load('0119-moe-form-yubin-WIKI103-ALL.pth')
    # model, optimizer, lr_scheduler, train_loader, val_loader= accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    model, optimizer, lr_scheduler, train_loader, val_loader,val_loader1, val_loader2= accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader,val_loader1,val_loader2)
    
    # module = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    o = 0

    steps = 0
    # accelerator.save_state('MODELS_0506/MOE-new-768ffns-wikipedia-large-1.5e-4-checkpoints%d'%steps)

    losses = []
    # for i, batch in enumerate(val_loader_domain1_ntk):
        
    #     batchl = {k: v[:1, :] for k, v in batch.items()}
    #     break
    for epoch in range(num_epochs):

        model.train()
        
        # RS = [{0:0,1:0,2:0,3:0} for i in range(12)]
        for i, batch in enumerate(train_loader):
            steps += 1
            # _, _, _ = get_ntk_by_layer(model,val_loader_domain1_ntk,steps)
            # _, _, _ = get_ntk_by_layer(model,val_loader_domain1_ntk,steps,"legal",batchl)

            # step = epoch * len(train_loader) + i
            loss, _ ,_,_,ROUTES= model(**batch)
            # print(ROUTES)
            # for p in range(len(R)):
            #     for j in R[p]:
            #         RS[p][j.item()]+=1
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            losses.append(accelerator.gather(loss.repeat(config.batch_size)))

            # if step % 500 == 0:
            #     if accelerator.is_local_main_process:
            #         loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
            #         writer.add_scalar('loss_train', loss_train, step)
            if steps%10 == 0:
                print("steps %d"%steps)
                for l in range(12):
                    ROUTES_PRO = {}
                    for e in range(config.num_experts):
                        equals_e = torch.eq(ROUTES[l], e)
                        ROUTES_PRO[e] = round(torch.sum(equals_e).item()/len(ROUTES[l]),2)
                    print("layer %3d    "%l,ROUTES_PRO)
                get_centers_radius(model, val_loader, accelerator,steps)

            if steps%100 ==0:
                loss_train = torch.mean(torch.cat(losses)[:len(losses)])
                losses=[]
                loss_valid = validate(model, val_loader, accelerator)
                loss_valid1 = validate(model, val_loader1, accelerator)
                loss_valid2 = validate(model, val_loader2, accelerator)
                # loss_valid3 = validate(model, val_loader3, accelerator)

                # ahead_loss_valid = validate(model, ahead_val_loader, accelerator)
                # loss_test = validate(model, test_loader, accelerator)
                accelerator.print(f'Epoch:{epoch} ({i} Updates), steps:{steps}, Train Loss: {loss_train}, Valid Loss: {loss_valid}')
                # accelerator.print(RS)
                if accelerator.is_local_main_process:
                    writer.add_scalar('perplexity_train_epoch', loss_train, steps)
                    writer.add_scalar('perplexity_valid', loss_valid, steps)
                    writer.add_scalar('leagl_perplexity_valid', loss_valid1, steps)

                    writer.add_scalar('pubmed_perplexity_valid', loss_valid2, steps)
                    # writer.add_scalar('general_perplexity_valid', loss_valid3, steps)

                    # writer.add_scalar('perplexity_valid_ahead', ahead_loss_valid, steps)
                    # writer.add_scalar('perplexity_test', loss_test, epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], steps)

            # if steps%7000 == 0:
                # loss_valid_f = validate(model, val_loader_f, accelerator)
                # if accelerator.is_local_main_process:
                #     writer.add_scalar('more_perplexity_valid', loss_valid_f, steps)
                # accelerator.save_state('MODELS_0514/MOE-768ffns-mixed-small-1.5e-4-checkpoints%d'%steps)
                
                # torch.save(model, 'MODELS_0315/0330-MoE-768ffns-longtailed-both-2e-4-NTK-checkpoints%d.pth'%steps)
            # if steps%2000 == 0 and steps>1000:
            #     _, _, _,o = get_ntk_by_layer(model,val_loader_domain1_ntk,steps,"legal",batchl,o)
            #     _, _, _,o = get_ntk_by_layer(model,val_loader_domain2_ntk,steps,"bio",batchl,o)
    
    
    # torch.save(model, 'MODELS_0315/0330-MoE-768ffns-longtailed-both-2e-4-NTK.pth')
    accelerator.save_state('/home/jxzhou/PLM_PER/BERT-CL-main/MOE_HUAWEI/MOE-MODELS')

def validate(model: BertForMLM, val_loader, accelerator):
    losses = []
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            loss, loss_dict,_,_,_= model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity


def get_centers_radius(model, val_loader, accelerator,steps):
    FFNS = [[[]for j in range(config.num_experts)]for l in range(12)]
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            if i%100 == 0:
                _, _,ffns,_,routes= model(**batch)
                # print(len(routes[0]))
                for l in range(12):
                    # inde = [torch.eq(routes[l], i).nonzero(as_tuple=True)[0] for i in range(config.num_experts)]
                    # print(inde)
                    for e in range(config.num_experts):
                        # print(routes)
                        
                        # inde = [index for index in routes[l] if element == e]
                        # print(ffns[l].view(-1,768)[routes[l] == e])
                        FFNS[l][e].append(ffns[l].view(-1,768)[routes[l] == e])
                    
    CENTERS = []
    RADIUS = []
    O_CENTERS = []
    O_RADIUS = []
    CLUSTERS_IDS = []
    for l in range(12):
        for e in range(config.num_experts):
            FFNS[l][e]=torch.cat(FFNS[l][e])
            # print(l,e)
        
        O_CENTERS.append([torch.mean(FFNS[l][j]) for j in range(config.num_experts)])
        O_RADIUS.append([torch.std(FFNS[l][j]) for j in range(config.num_experts)])
        FFNS[l] = torch.cat(FFNS[l]).cpu().numpy()
        kmeans = KMeans(n_clusters=config.num_experts, n_init="auto").fit(FFNS[l])

        # 获取质心
        labels = kmeans.labels_

        # 统计各个簇的数量
        cluster_counts = {}
        for label in labels:
            if label in cluster_counts:
                cluster_counts[label] += 1
            else:
                cluster_counts[label] = 1
        CLUSTERS_IDS.append(cluster_counts)
        centers = kmeans.cluster_centers_
        CENTERS.append(centers)

        # 计算每个簇的半径
        radii = []
        for i in range(kmeans.n_clusters):
            # 获取属于第i个簇的所有数据点的索引
            indices = np.where(kmeans.labels_ == i)[0]
            # 计算这些数据点到质心的距离
            distances = np.linalg.norm(FFNS[l][indices] - centers[i], axis=1)
            # 计算平均距离
            radius = np.mean(distances)
            radii.append(radius)
        RADIUS.append(radii)

    # print("KMEANS: ",CENTERS,RADIUS)
    # print("ORIGIN: ",O_CENTERS,O_RADIUS)
    torch.save(O_CENTERS,"MOE_FOR_HUAWEI_PTH/O_CENTERS-ATT-steps%d.pth"%steps)
    torch.save(O_RADIUS,"MOE_FOR_HUAWEI_PTH/O_RADIUS-ATT-steps%d.pth"%steps)

    torch.save(CENTERS,"MOE_FOR_HUAWEI_PTH/CENTERS-ATT-steps%d.pth"%steps)
    torch.save(RADIUS,"MOE_FOR_HUAWEI_PTH/RADIUS-ATT-steps%d.pth"%steps)
    torch.save(CLUSTERS_IDS,"MOE_FOR_HUAWEI_PTH/CLUSTER-IDS-ATT-steps%d.pth"%steps)

    
    return 0

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




def get_ntk_by_layer(model,loader,steps,doamin,longtailed_data,d):

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
                


                _, logits, _, _ = model(**batch)

                for i in range(len(logits)):
                    # print(i)
                    logits[i:i+1].backward(torch.ones_like(logits[i:i+1]), create_graph=True)
                    this_grad.append(param.grad.detach().reshape(-1).clone())
                    model.zero_grad()
                    torch.cuda.empty_cache()

                
                
                _, logits, _, _= model(**longtailed_data)

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

    plt.figure(d,dpi=120)
    sns.heatmap(data=lambdamax_layers[:,:],
                vmin=1e11,
                vmax=1e16,
                cmap=plt.get_cmap('Blues')
        )
    plt.title('LAMBDAmax_IN_HEADS_EVERY_LAYERS') 
    plt.savefig('NTKS-0330-MOE/NTK-lambdamax-%dsteps-%s.png'%(steps,doamin)) 
    d+=1

    plt.figure(d,dpi=120)
    sns.heatmap(data=longtailed_NTK[:,:],
                vmin=-1,
                vmax=1,
                cmap=plt.get_cmap('Blues')
        )
    plt.title('PROJECTIONS_IN_HEADS_EVERY_LAYERS') 
    plt.savefig('NTKS-0330-MOE/NTK-projection-%dsteps-%s.png'%(steps,doamin)) 
    d+=1
    plt.figure(d,dpi=120)
    sns.heatmap(data=longtailed_sim[:,:],
                vmin=-1,
                vmax=1,
                cmap=plt.get_cmap('Blues')
        )
    plt.title('SIMILARITY_IN_HEADS_EVERY_LAYERS') 
    plt.savefig('NTKS-0330-MOE/NTK-similarity-%dsteps-%s.png'%(steps,doamin)) 
    d+=1
    print(steps,"finished")
    return NTKs, NTK,lambdamax_layers,d
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters_detailed(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params

if __name__ == "__main__":
    seed = 28
    set_seed(seed)
    config = BertConfig.from_json_file('config/MoMoE.json')
    # model = BertForMLM_every2layers(config)
    model = BertForMLM(config)

    # torch.cuda.set_device(4)
    epoch = 1
    LEGAL = 1000000
    # dataset = MoMoE_longtailed(config = config,KKK=LEGAL)
    dataset = MoE_huawei(config = config,KKK=LEGAL)

    # # ahead_dataset = Wikipedia_Wids(config = config)
    torch.cuda.set_device(2)
    device = torch.device("cuda:2")
    # # dataset = PubMedForLM(config)

    # load_checkpoint_and_dispatch(model, '/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0514/MOE-768ffns-mixed-small-1.5e-4',device_map={"":device})
    # model = torch.load(os.path.join('/home/jxzhou/PLM_PER/BERT-CL-main/MODELS_0514/MOE-768ffns-mixed-small-1.5e-4', 'pytorch_model.bin'))
    # print(model)

    train(model, dataset, epoch)
    # total_params, trainable_params, non_trainable_params = count_parameters_detailed(model)
    # print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")
    # print(f"Non-trainable parameters: {non_trainable_params}")