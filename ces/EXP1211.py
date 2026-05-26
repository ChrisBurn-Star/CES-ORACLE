import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset_new import RestaurantforLM_1103, MixedData, MixedData_stage1, ACLForLM,old_MixedData_after_stage1, Mixdata_1103, Wikitext,ACLForLM_1103,Mixdata_1115,Review_1103,MixedData_1211
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim

import torch
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D


def show_project(inputs,inputs2,L=0):
    

    l = 5
    
    data = torch.mean(inputs[l-1],1)
    
    data2 = torch.mean(inputs2[l],1)

    U, S, V = torch.svd(data)
    A = data2
    projection_matrix0 = U.mm(U.t().mm(data)).cpu().numpy()
    projection_matrix0 = np.mean(projection_matrix0,0)
    projection_matrix0 = F.normalize(torch.tensor(projection_matrix0), p=2, dim=0)
    projection_matrix = U.mm(U.t().mm(A)).cpu().numpy()
    projection_matrix = np.mean(projection_matrix,0)
    projection_matrix = F.normalize(torch.tensor(projection_matrix), p=2, dim=0)
    # print(projection_matrix.shape)
    plt.figure(0)
    plt.plot(projection_matrix0.cpu().numpy(),label="inputs",alpha = 0.5 )
    plt.plot(projection_matrix.cpu().numpy(),label="attention",alpha = 0.5 )
    plt.xlabel('Dimension')
    plt.ylabel('Projection')
    plt.title('Projection per Dimension')
    plt.legend()
    plt.savefig("1215-png/projection.png")



    if L:
        data = []
        for d in inputs:
            data.append(d[0:64,:,:])
        data = torch.cat(data)
        # print(data.shape)
        data = torch.mean(data,1)
        U, S, V = torch.svd(data)
        for l in range(6):
            A = torch.mean(inputs[l],1)
            # U, S, V = torch.svd(A)

            # 重构对角矩阵
            # Sigma = torch.zeros((A.shape[0], A.shape[1]), dtype=torch.float)
            # Sigma[:min(A.shape[0], A.shape[1]), :] = torch.diag(S)

            # 计算投影矩阵
            # projection_matrix = torch.mm(torch.mm(U.to('cuda'), Sigma.to('cuda')), V.t().to('cuda')).cpu().numpy()
            # projection_matrix = np.mean(projection_matrix,0)
            

            projection_matrix = U.mm(U.t().mm(A)).cpu().numpy()
            projection_matrix = np.mean(projection_matrix,0)
            # print(projection_matrix.shape)
            plt.figure(0)
            plt.plot(projection_matrix,label=l,alpha = 0.5 )
            plt.xlabel('Dimension')
            plt.ylabel('Projection')
            plt.title('Projection per Dimension')
            plt.legend()
            plt.savefig("1215-png/projection1.png")

        for l in range(6,12):
            A = torch.mean(inputs[l],1)
            # U, S, V = torch.svd(A)

            # 重构对角矩阵
            # Sigma = torch.zeros((A.shape[0], A.shape[1]), dtype=torch.float)
            # Sigma[:min(A.shape[0], A.shape[1]), :] = torch.diag(S)

            # 计算投影矩阵
            # projection_matrix = torch.mm(torch.mm(U.to('cuda'), Sigma.to('cuda')), V.t().to('cuda')).cpu().numpy()
            # projection_matrix = np.mean(projection_matrix,0)
            projection_matrix = U.mm(U.t().mm(A)).cpu().numpy()
            projection_matrix = np.mean(projection_matrix,0)
            plt.figure(1)
            plt.plot(projection_matrix,label=l,alpha = 0.5 )
            plt.xlabel('Dimension')
            plt.ylabel('Projection')
            plt.title('Projection per Dimension')
            plt.legend()
            plt.savefig("1215-png/projection2.png")

def show_token_pca(inputs):
    l = 5
    x = [[] for i in range(128)]
    pca = PCA(n_components=2)
    y = []
    z = []
    for i in range(len(x)):
        transformed_data = pca.fit_transform(inputs[l][:,i,:].cpu().numpy())
        # print(transformed_data)
        x[i]=transformed_data
        # y.append(np.mean(transformed_data[:,0]))
        # z.append(np.mean(transformed_data[:,1]))
    x = np.array(x)
    print(x.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(128):

        for j in range(x.shape[1]):
            ax.scatter(i,x[i,j,0],x[i,j,1])


    ax.set_title('3D Scatter Plot')
    ax.set_xlabel('tokens')
    ax.set_ylabel('PCA1')
    ax.set_zlabel('PCA2')

    plt.savefig("1215-png/tokens.png")


def show_cos(i_o_a, o_o_a, o_o_f, o_w_r, o_o_l):

    # l=9
    for l in range(1,10):
        in_o_a = torch.mean(i_o_a[l-1],1)
        ou_o_a = torch.mean(o_o_a[l],1)

        ou_o_f = torch.mean(o_o_f[l],1)
        ou_w_r = torch.mean(o_w_r[l],1)
        eigenvalues1 = torch.var(in_o_a,dim=0)
        eigenvalues2 = torch.var(ou_o_a,dim=0)
        eigenvalues3 = torch.var(ou_o_f,dim=0)
        eigenvalues4 = torch.var(ou_w_r,dim=0)

        n_e1 = F.normalize(eigenvalues1, p=2, dim=0)
        n_e2 = F.normalize(eigenvalues2, p=2, dim=0)
        n_e3 = F.normalize(eigenvalues3, p=2, dim=0)
        n_e4 = F.normalize(eigenvalues4, p=2, dim=0)


        similarity1 = F.cosine_similarity(n_e1,n_e2, dim=0).item()
        similarity2 = F.cosine_similarity(n_e2,n_e3, dim=0).item()
        similarity3 = F.cosine_similarity(n_e3,n_e4, dim=0).item()

        plt.figure(0)
        plt.plot(["OUT_OF_ATT - IN_OF_ATT", "OUT_OF_FFN - IN_OF_FFN", "OUT_WITH_RES - OUT_OF_FFN"], [similarity1,similarity2,similarity3], label=l)
        plt.xlabel('Dimension')
        plt.ylabel('Variance')
        plt.title('Variance per Dimension')
        plt.legend()
        plt.savefig("1214-png/"+"cosine_var.png")



def show_engis(inputs,inputs2):

    K = 15
    for i in range(6):
        # input = torch.mean(inputs[i],1)
        input = inputs[i].view(-1,768)
        X = input
        # input = torch.matmul(X.t(),input)
        # eigenvalues0, _ = torch.linalg.eig(input)
        U, S, V = torch.svd(X)
        # indices = torch.argsort(eigenvalues0)
        eigenvalues = F.normalize(S, p=2, dim=0)
        non_zero_elements = eigenvalues[eigenvalues != 0]
        print(len(non_zero_elements[0:K]))
        GM = (torch.prod(non_zero_elements[0:K]))**(1/(len(non_zero_elements[0:K])))
        HM = (torch.sum(torch.reciprocal(non_zero_elements[0:K])))**(-1)

        print(i,GM,HM)




        fig100 = plt.figure(100)
        plt.scatter(i,GM.item(),c="r")
        plt.scatter(i,HM.item(),c="b")
        plt.xlabel('LAYERS')
        plt.ylabel('GM/HM')
        plt.title('GM/HM CHANGES IN LAYERS')
        plt.legend()
        fig100.savefig("1214-png/GM-HM1.png")


        plt.figure(10)
        plt.plot(eigenvalues.cpu().numpy(),label=i )
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalues')
        plt.title('Eigenvalues per Dimension')
        plt.ylim(0, 0.1)  #
        plt.legend()
        plt.savefig("1214-png/attention_output_lambdas100.png")

        plt.figure(11)
        plt.plot(eigenvalues[0:192].cpu().numpy(),label=i )
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalues')
        plt.title('Eigenvalues per Dimension')
        plt.legend()
        plt.savefig("1214-png/attention_output_lambdas192.png")


        plt.figure(12)
        plt.plot(eigenvalues[192:384].cpu().numpy(),label=i )
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalues')
        plt.title('Eigenvalues per Dimension')
        plt.legend()
        plt.savefig("1214-png/attention_output_lambdas384.png")

        plt.figure(13)
        plt.plot(eigenvalues[384:576].cpu().numpy(),label=i )
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalues')
        plt.title('Eigenvalues per Dimension')
        plt.legend()
        plt.savefig("1214-png/attention_output_lambdas576.png")

        plt.figure(14)
        plt.plot(eigenvalues[576:768].cpu().numpy(),label=i )
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalues')
        plt.title('Eigenvalues per Dimension')
        plt.legend()
        plt.savefig("1214-png/attention_output_lambdas768.png")
    for i in range(6,12):
        # input = torch.mean(inputs[i],1)
        input = inputs[i].view(-1,768)
        X = input
        # input = torch.matmul(X.t(),input)
        # eigenvalues0, _ = torch.linalg.eig(input)
        U, S, V = torch.svd(X)
        # indices = torch.argsort(eigenvalues0)
        eigenvalues = F.normalize(S, p=2, dim=0)
        non_zero_elements = eigenvalues[eigenvalues != 0]
        GM = (torch.prod(non_zero_elements[0:K]))**(1/(len(non_zero_elements[0:K])))
        HM = (torch.sum(torch.reciprocal(non_zero_elements[0:K])))**(-1)

        print(i,GM,HM)


        fig200 = plt.figure(200)
        plt.scatter(i,GM.item(),c="r")
        plt.scatter(i,HM.item(),c="b")
        plt.xlabel('LAYERS')
        plt.ylabel('GM/HM')
        plt.title('GM/HM CHANGES IN LAYERS')
        
        plt.legend()
        fig200.savefig("1214-png/GM-HM2.png")

        plt.figure(0)
        plt.plot(eigenvalues.cpu().numpy(),label=i )
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalues')
        plt.title('Eigenvalues per Dimension')
        plt.ylim(0, 0.1)
        plt.legend()
        plt.savefig("1215-png/attention_output_lambdas100.png")

        plt.figure(1)
        plt.plot(eigenvalues[0:192].cpu().numpy(),label=i )
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalues')
        plt.title('Eigenvalues per Dimension')
        plt.legend()
        plt.savefig("1215-png/attention_output_lambdas192.png")


        plt.figure(2)
        plt.plot(eigenvalues[192:384].cpu().numpy(),label=i )
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalues')
        plt.title('Eigenvalues per Dimension')
        plt.legend()
        plt.savefig("1215-png/attention_output_lambdas384.png")

        plt.figure(3)
        plt.plot(eigenvalues[384:576].cpu().numpy(),label=i )
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalues')
        plt.title('Eigenvalues per Dimension')
        plt.legend()
        plt.savefig("1215-png/attention_output_lambdas576.png")

        plt.figure(4)
        plt.plot(eigenvalues[576:768].cpu().numpy(),label=i )
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalues')
        plt.title('Eigenvalues per Dimension')
        plt.legend()
        plt.savefig("1215-png/attention_output_lambdas768.png")


def show_var(att_inputs,outputs,loa):
    l = 10##11>=l>=1  
    output = torch.mean(outputs[l-1],1)
    input = torch.mean(att_inputs[l],1)
    fin_out = torch.mean(outputs[l],1)
    # input = inputs[i].view(-1,768)

    eigenvalues = torch.var(input,dim=0)
    eigenvalues2 = torch.var(output,dim=0)
    eigenvalues3 = torch.var(fin_out,dim=0)
    # indices = torch.argsort(eigenvalues0)
    # eigenvalues = eigenvalues0[indices]
    plt.figure(0)
    # plt.plot(eigenvalues.cpu().numpy(),label="affn_outputs(without res)" ,alpha=0.5)
    plt.bar([i for i in range(768)],eigenvalues2.cpu().numpy() - np.mean(eigenvalues2.cpu().numpy()),label="ffn_input" ,alpha=0.5)
    
    plt.xlabel('Dimension')
    plt.ylabel('Variance')
    plt.title('Variance per Dimension')
    plt.legend()
    plt.savefig("1214-png/"+loa+"_varall.png")
    plt.figure(81)
    plt.bar([i for i in range(768)],eigenvalues3.cpu().numpy()-np.mean(eigenvalues3.cpu().numpy()),label="ffn_output" ,alpha=0.5)
    plt.xlabel('Dimension')
    plt.ylabel('Variance')
    plt.title('Variance per Dimension')
    plt.legend()
    plt.savefig("1214-png/"+loa+"_varall2.png")

    plt.figure(10)
    plt.plot(eigenvalues.cpu().numpy() - eigenvalues2.cpu().numpy() ,label="attention_outputs(without res) - layer_input" ,alpha=0.5)
    plt.plot(eigenvalues3.cpu().numpy() - eigenvalues2.cpu().numpy(),label="layer_output(with res) - layer_input" ,alpha=0.5)
    plt.xlabel('Dimension')
    plt.ylabel('Variance Difference')
    plt.title('Variance per Dimension')
    plt.legend()
    plt.savefig("1214-png/"+loa+"_vardiff.png")

    plt.figure(1)
    plt.plot(eigenvalues[0:192].cpu().numpy(),label="attention_outputs(without res)" ,alpha=0.5)
    plt.plot(eigenvalues2[0:192].cpu().numpy(),label="layer_input" ,alpha=0.5)
    plt.plot(eigenvalues3[0:192].cpu().numpy(),label="layer_output(with res)" ,alpha=0.5)
    plt.xlabel('Dimension')
    plt.ylabel('Variance')
    plt.title('Variance per Dimension')
    plt.legend()
    plt.savefig("1214-png/"+loa+"_var192.png")

    plt.figure(11)
    plt.plot(eigenvalues[0:192].cpu().numpy() - eigenvalues2[0:192].cpu().numpy() ,label="attention_outputs(without res) - layer_input" ,alpha=0.5)
    plt.plot(eigenvalues3[0:192].cpu().numpy() - eigenvalues2[0:192].cpu().numpy(),label="layer_output(with res) - layer_input" ,alpha=0.5)
    plt.xlabel('Dimension')
    plt.ylabel('Variance Difference')
    plt.title('Variance per Dimension')
    plt.legend()
    plt.savefig("1214-png/"+loa+"_vardiff192.png")

    plt.figure(2)
    plt.plot(eigenvalues[192:384].cpu().numpy(),label="attention_outputs(without res)" ,alpha=0.5)
    plt.plot(eigenvalues2[192:384].cpu().numpy(),label="layer_input" ,alpha=0.5)
    plt.plot(eigenvalues3[192:384].cpu().numpy(),label="layer_output(with res)" ,alpha=0.5)
    plt.xlabel('Dimension')
    plt.ylabel('Variance')
    plt.title('Variance per Dimension')
    plt.legend()
    plt.savefig("1214-png/"+loa+"_var384.png")

    plt.figure(12)
    plt.plot(eigenvalues[192:384].cpu().numpy() - eigenvalues2[192:384].cpu().numpy() ,label="attention_outputs(without res) - layer_input" ,alpha=0.5)
    plt.plot(eigenvalues3[192:384].cpu().numpy() - eigenvalues2[192:384].cpu().numpy(),label="layer_output(with res) - layer_input" ,alpha=0.5)
    plt.xlabel('Dimension')
    plt.ylabel('Variance Difference')
    plt.title('Variance per Dimension')
    plt.legend()
    plt.savefig("1214-png/"+loa+"_vardiff384.png")

    plt.figure(3)
    plt.plot(eigenvalues[384:576].cpu().numpy(),label="attention_outputs(without res)" ,alpha=0.5)
    plt.plot(eigenvalues2[384:576].cpu().numpy(),label="layer_input" ,alpha=0.5)
    plt.plot(eigenvalues3[384:576].cpu().numpy(),label="layer_output(with res)" ,alpha=0.5)
    plt.xlabel('Dimension')
    plt.ylabel('Variance')
    plt.title('Variance per Dimension')
    plt.legend()
    plt.savefig("1214-png/"+loa+"_var576.png")

    plt.figure(13)
    plt.plot(eigenvalues[384:576].cpu().numpy() - eigenvalues2[384:576].cpu().numpy() ,label="attention_outputs(without res) - layer_input" ,alpha=0.5)
    plt.plot(eigenvalues3[384:576].cpu().numpy() - eigenvalues2[384:576].cpu().numpy(),label="layer_output(with res) - layer_input" ,alpha=0.5)
    plt.xlabel('Dimension')
    plt.ylabel('Variance Difference')
    plt.title('Variance per Dimension')
    plt.legend()
    plt.savefig("1214-png/"+loa+"_vardiff576.png")

    plt.figure(4)
    plt.plot(eigenvalues[576:768].cpu().numpy(),label="attention_outputs(without res)" ,alpha=0.5)
    plt.plot(eigenvalues2[576:768].cpu().numpy(),label="layer_input" ,alpha=0.5)
    plt.plot(eigenvalues3[576:768].cpu().numpy(),label="layer_output(with res)" ,alpha=0.5)
    plt.xlabel('Dimension')
    plt.ylabel('Variance')
    plt.title('Variance per Dimension')
    plt.legend()
    plt.savefig("1214-png/"+loa+"_var768.png")

    plt.figure(14)
    plt.plot(eigenvalues[576:768].cpu().numpy() - eigenvalues2[576:768].cpu().numpy() ,label="attention_outputs(without res) - layer_input" ,alpha=0.5)
    plt.plot(eigenvalues3[576:768].cpu().numpy() - eigenvalues2[576:768].cpu().numpy(),label="layer_output(with res) - layer_input" ,alpha=0.5)
    plt.xlabel('Dimension')
    plt.ylabel('Variance Difference')
    plt.title('Variance per Dimension')
    plt.legend()
    plt.savefig("1214-png/"+loa+"_vardiff768.png")

    # for i in range(len(inputs)):
    #     input = torch.mean(inputs[i],1)
    #     # input = inputs[i].view(-1,768)

    #     eigenvalues = torch.var(input,dim=0)
    #     # indices = torch.argsort(eigenvalues0)
    #     # eigenvalues = eigenvalues0[indices]
    #     plt.figure(0)
    #     plt.plot(eigenvalues.cpu().numpy(),label=i )
    #     plt.xlabel('Dimension')
    #     plt.ylabel('Variance')
    #     plt.title('Variance per Dimension')
    #     plt.legend()
    #     plt.savefig("1214-png/"+loa+"_varall.png")


    #     plt.figure(1)
    #     plt.plot(eigenvalues[0:192].cpu().numpy(),label=i )
    #     plt.xlabel('Dimension')
    #     plt.ylabel('Variance')
    #     plt.title('Variance per Dimension')
    #     plt.legend()
    #     plt.savefig("1214-png/"+loa+"_var192.png")


    #     plt.figure(2)
    #     plt.plot(eigenvalues[192:384].cpu().numpy(),label=i )
    #     plt.xlabel('Dimension')
    #     plt.ylabel('Variance')
    #     plt.title('Variance per Dimension')
    #     plt.legend()
    #     plt.savefig("1214-png/"+loa+"_var384.png")

    #     plt.figure(3)
    #     plt.plot(eigenvalues[384:576].cpu().numpy(),label=i )
    #     plt.xlabel('Dimension')
    #     plt.ylabel('Variance')
    #     plt.title('Variance per Dimension')
    #     plt.legend()
    #     plt.savefig("1214-png/"+loa+"_var576.png")

    #     plt.figure(4)
    #     plt.plot(eigenvalues[576:768].cpu().numpy(),label=i )
    #     plt.xlabel('Dimension')
    #     plt.ylabel('Variance')
    #     plt.title('Variance per Dimension')
    #     plt.legend()
    #     plt.savefig("1214-png/"+loa+"_var768.png")

    
def get_rep_distribute(datas,l,m):
    for i in range(len(datas)):
        datas[i] = torch.mean(datas[i],1)
        # datas[i] = torch.mean(datas[i],0)
        # datas[i] = datas[i].cpu().numpy()
    # X=[i for i in range(768)]
    plt.figure(0)
    for i in range(len(datas)):
        # tensor = abs(datas[i])
        tensor = torch.var(datas[i], dim = 0)
        # median = np.mean(tensor)
        # tensor[tensor<=median] = 0
        top_values, top_indices = torch.topk(tensor, k=5)

        
        plt.bar(top_indices.cpu().numpy(),[i for j in range(len(top_indices))],label = i, alpha = 0.5)
    plt.title('REP Visualization')
    plt.xlabel('MAX-VARIANCE Dimension')
    plt.ylabel('LYARES')
    plt.legend()
    F= "1215-png/REP_OF_LAYER.png"
    plt.savefig(F)

    # plt.figure(l+100)
    
    # plt.plot(X,abs(datas[1]-datas[0]),label = "difference")
    # plt.plot(X,abs(datas[0]),label = "after attention")
    # plt.title('REP Visualization')
    # plt.xlabel('Dimension')
    # plt.ylabel('VALUES')
    # plt.legend()
    # F= "1213-png%d/DIFF_REP_OF_%d_LAYER.png"%(m,l)
    # plt.savefig(F)
    # plt.figure(l+200)
    
    # plt.plot(X,abs(datas[4]-datas[3]),label = "difference")
    # plt.plot(X,abs(datas[3]),label = "after attention")
    # plt.title('REP Visualization')
    # plt.xlabel('Dimension')
    # plt.ylabel('VALUES')
    # plt.legend()
    # F= "1213-png%d/FFNDIFF_REP_OF_%d_LAYER.png"%(m,l)
    # plt.savefig(F)

def show_eignvalues(data,config,f,a,l):
    # data = data.view(-1, config.seq_len*config.hidden_size)
    data = torch.mean(data,1)
    data = data.cpu().numpy()
    pca = PCA()
    pca.fit(data)

    # 获取所有特征值
    eigenvalues = pca.explained_variance_

    plt.figure(a)
    plt.plot(eigenvalues,label = l)
    plt.title('Eigenvalues Visualization')
    plt.xlabel('Original Dimension')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.xlim(0, 100)  #
    plt.ylim(0, max(eigenvalues) * 1.1)
    plt.legend()  #
    plt.savefig(f)



def show_PCA_cluster(datas):
    data = []
    for d in datas:
        data.append(d[0:64,:,:])
    data = torch.cat(data)
    # print(data.shape)
    data = torch.mean(data,1)
    pca = PCA(n_components=2)
    pca.fit(data.cpu().numpy())
    for i in range(len(datas)):
        transformed_datak = pca.transform(torch.mean(datas[i][0:64,:,:],1).cpu().numpy())
        plt.figure(i)
        plt.scatter(transformed_datak[:, 0], transformed_datak[:, 1], alpha=0.5)
        plt.title('PCA Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        # plt.xlim(-30, 30)  #
        # plt.ylim(-30, 30)  #
        F= "1215-png/PCA_TOKENS_layer%d"%i+"seqloca"+".png"
        plt.savefig(F)

def show_cluster(data0,config,f,l):
    # data = np.random.random((1000, 128, 768))  # 这里用随机数据代替
    # data = data.view(-1, config.seq_len*config.hidden_size)
    # data = torch.mean(data,1)
    # for k in range(0,128,50):
        # print(data0.shape)
    data = data0.view(-1,768)
    data = data.cpu().numpy()
    # print(data.shape)
    reshaped_data =  data# 重新塑形为1000*98304的形状

    # tsne = TSNE(n_components=2)  # 选择降维到2维
    # transformed_data = tsne.fit_transform(reshaped_data)

    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(data)

    t = 64
    # for i in range(12):

    #     # pca = PCA(n_components=data[:,i*t:(i+1)*t].shape[1])
    #     # pca.fit(data[:,i*t:(i+1)*t])

    #     # # 获取信息量最大的维度索引
    #     # max_variance_index = torch.argmax(torch.tensor(pca.explained_variance_))
    #     # print(l, t*i,max_variance_index.item())
    #     transformed_datak = pca.fit_transform(data[:,i*t:(i+1)*t])
    #     plt.figure()
    #     plt.scatter(transformed_datak[:, 0], transformed_datak[:, 1], alpha=0.5)
    #     plt.title('PCA Visualization')
    #     plt.xlabel('Principal Component 1')
    #     plt.ylabel('Principal Component 2')
    #     # plt.xlim(-30, 30)  #
    #     # plt.ylim(-30, 30)  #
    #     F= f+"seqloca_"+str(k)+"_"+str(t*i)+".png"
    #     plt.savefig(F)
    pca = PCA(n_components=2)
    transformed_datak = pca.fit_transform(data)
    plt.figure()
    plt.scatter(transformed_datak[:, 0], transformed_datak[:, 1], alpha=0.5)
    plt.title('PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # plt.xlim(-30, 30)  #
    # plt.ylim(-30, 30)  #
    F= f+"seqloca"+".png"
    plt.savefig(F)
    
    # mds = MDS(n_components=2)
    # transformed_data = mds.fit_transform(data)
    # plt.figure()
    # plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    # plt.title('MDS Visualization')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.figure()
    # plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5)
    # plt.title('t-SNE Visualization')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.xlim(-30, 30)  #
    # plt.ylim(-30, 30)  #
    # plt.savefig(f)

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
            loss, _, _, _, _, _,_,_ = model(**batch)
            # loss, _, _, _, _ = model(**batch)

        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity


def get_gradient_norms(model):
    """Utility function to get gradient norms of a model."""
    return [param.grad.norm().item() for param in model.parameters() if param.grad is not None]

def show_clustering(model,dataset,device,config):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader 
    OUTPUTS_OF_ATTEN = [[] for i in range(12)]
    OUTPUTS_AFTER_ATTEN = [[] for i in range(12)]
    OUTPUTS_OF_FFN = [[] for i in range(12)]
    OUTPUTS_AFTER_FFN = [[] for i in range(12)]
    FEB_OUTS_AFTER_ATT = [[] for i in range(12)]
    FEB_OUTS_AFTER_FFN = [[] for i in range(12)]
    EMBEDDINGS = []
    for i, batch in enumerate(train_loader):  
        batch = {key: tensor.to(device) for key, tensor in batch.items()}      
        with torch.no_grad():
            if i%67 ==0:
                print(i)
                # _, _, ATTOUTS,ATTOS, FFNOS,OUTSA,FEB_ATT, FEB_FFN = model(**batch)
                batch = {key: tensor.to(device) for key, tensor in batch.items()}

                _, _, ATTOUTS, ATTOS, FFNOS, OUTSA,FEB_ATT, FEB_FFN = model(**batch)
                # for a in range(len(ATTOUTS)):
                #     OUTPUTS_OF_ATTEN[a].append(ATTOUTS[a])
                for a in range(len(ATTOS)):
                    OUTPUTS_AFTER_ATTEN[a].append(ATTOS[a])
                for a in range(len(FFNOS)):
                    OUTPUTS_OF_FFN[a].append(FFNOS[a])
                for a in range(len(OUTSA[1:])):
                    OUTPUTS_AFTER_FFN[a].append(OUTSA[a+1])
                # # for a in range(len(FEB_ATT)):
                # #     FEB_OUTS_AFTER_ATT[a].append(FEB_ATT[a])
                # for a in range(len(FEB_FFN)):
                #     FEB_OUTS_AFTER_FFN[a].append(FEB_FFN[a])
                # # EMBEDDINGS.append(OUTSA[0])
    # for a in range(len(OUTPUTS_OF_ATTEN)):
    #     OUTPUTS_OF_ATTEN[a]=torch.cat(OUTPUTS_OF_ATTEN[a])
    for a in range(len(OUTPUTS_AFTER_ATTEN)):
        OUTPUTS_AFTER_ATTEN[a]=torch.cat(OUTPUTS_AFTER_ATTEN[a])
    for a in range(len(OUTPUTS_OF_FFN)):
        OUTPUTS_OF_FFN[a]=torch.cat(OUTPUTS_OF_FFN[a])
    for a in range(len(OUTPUTS_AFTER_FFN)):
        OUTPUTS_AFTER_FFN[a]=torch.cat(OUTPUTS_AFTER_FFN[a])
    # # for a in range(len(FEB_OUTS_AFTER_ATT)):
    # #     FEB_OUTS_AFTER_ATT[a]=torch.cat(FEB_OUTS_AFTER_ATT[a])
    # for a in range(len(FEB_OUTS_AFTER_FFN)):
    #     FEB_OUTS_AFTER_FFN[a]=torch.cat(FEB_OUTS_AFTER_FFN[a])
    
    # EMBEDDINGS=torch.cat(EMBEDDINGS)
    
    # f = "1213-png4/MDS_rep_EMBEDDINGS"+"Blayer%d"%i
    # show_cluster(EMBEDDINGS,config,f)


    # for i in range(len(OUTPUTS_OF_ATTEN)):
    #     f = "1215-png/PCA_rep_OUTPUTS_OF_ATTEN"+"layer%d"%i
    #     show_cluster(OUTPUTS_OF_ATTEN[i],config,f,i)
    # for i in range(len(OUTPUTS_AFTER_ATTEN)):
    #     f = "1213-png4/PCA_rep_OUTPUTS_AFTER_ATTEN"+"layer%d"%i
    #     show_cluster(OUTPUTS_AFTER_ATTEN[i],config,f)
    # for i in range(len(OUTPUTS_OF_FFN)):
    #     f = "1213-png4/PCA_rep_OUTPUTS_OF_FFN"+"layer%d"%i
    #     show_cluster(OUTPUTS_OF_FFN[i],config,f)
    # for i in range(len(OUTPUTS_AFTER_FFN)):
    #     f = "1213-png4/PCA_rep_OUTPUTS_AFTER_FFN"+"layer%d"%i
    #     show_cluster(OUTPUTS_AFTER_FFN[i],config,f)
    # for i in range(len(FEB_OUTS_AFTER_ATT)):
    #     f = "1213-png4/PCA_rep_FEB_OUTS_AFTER_ATT"+"layer%d"%i
    #     show_cluster(FEB_OUTS_AFTER_ATT[i],config,f)
    # for i in range(len(FEB_OUTS_AFTER_FFN)):
    #     f = "1213-png4/PCA_rep_FEB_OUTS_AFTER_FFN"+"layer%d"%i
    #     show_cluster(FEB_OUTS_AFTER_FFN[i],config,f)





    # for i in range(len(OUTPUTS_OF_ATTEN)):
    #     f = "1213-png1/eignvalues_OUTPUTS_OF_ATTEN"+"layer%d"%i
    #     show_eignvalues(OUTPUTS_OF_ATTEN[i],config,f,i,"1")
    # for i in range(len(OUTPUTS_AFTER_ATTEN)):
    #     f = "1213-png1/eignvalues_OUTPUTS_AFTER_ATTEN"+"layer%d"%i
    #     show_eignvalues(OUTPUTS_AFTER_ATTEN[i],config,f,i,"3")
    # for i in range(len(OUTPUTS_OF_FFN)):
    #     f = "1213-png1/eignvalues_OUTPUTS_OF_FFN"+"layer%d"%i
    #     show_eignvalues(OUTPUTS_OF_FFN[i],config,f,i,"4")
    # for i in range(len(OUTPUTS_AFTER_FFN)):
    #     f = "1213-png1/eignvalues_OUTPUTS_AFTER_FFN"+"layer%d"%i
    #     show_eignvalues(OUTPUTS_AFTER_FFN[i],config,f,i,"6")
    # for i in range(len(FEB_OUTS_AFTER_ATT)):
    #     f = "1213-png1/eignvalues_FEB_OUTS_AFTER_ATT"+"layer%d"%i
    #     show_eignvalues(FEB_OUTS_AFTER_ATT[i],config,f,i,"2")
    # for i in range(len(FEB_OUTS_AFTER_FFN)):
    #     f = "1213-png1/eignvalues_"+"layer%d"%i
    #     show_eignvalues(FEB_OUTS_AFTER_FFN[i],config,f,i,"5")
    # for l in range(12):
    #     REPS = []
    #     # REPS.append(OUTPUTS_OF_ATTEN[l])
    #     # REPS.append(FEB_OUTS_AFTER_ATT[l])
    #     # REPS.append(OUTPUTS_AFTER_ATTEN[l])

    #     # REPS.append(OUTPUTS_OF_FFN[l])
    #     # REPS.append(FEB_OUTS_AFTER_FFN[l])
    #     REPS.append(OUTPUTS_AFTER_FFN[l])
    get_rep_distribute(OUTPUTS_AFTER_FFN,0,2)
    # show_var(OUTPUTS_OF_ATTEN,"OUTPUTS_OF_ATTEN")
    # show_var(OUTPUTS_AFTER_ATTEN,OUTPUTS_OF_FFN,"changes")
        
    # show_cos(OUTPUTS_AFTER_FFN, OUTPUTS_OF_ATTEN, OUTPUTS_OF_FFN, FEB_OUTS_AFTER_FFN, OUTPUTS_AFTER_FFN)
        
    # show_engis(OUTPUTS_OF_ATTEN)
    # show_PCA_cluster(OUTPUTS_OF_ATTEN)
    # show_project(OUTPUTS_AFTER_FFN,OUTPUTS_OF_ATTEN,0)
        
    # show_token_pca(OUTPUTS_OF_ATTEN)
def train(model, num_epochs, dataset, device):
    # train_loader, val_loader, test_loader = dataset.train_loader, dataset.val_loader, dataset.test_loader
    # model = torch.load('1127-bert-only.pth')
    train_loader, val_loader = dataset.train_loader, dataset.val_loader

    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    accelerator = Accelerator()
    writer = SummaryWriter('EXP1211-A')
    EXP_EPOCH = [i for i in range(0,num_epochs,10)]
    EXP_STEP = 50
    num_updates = num_epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        
        """train origin bert (MLM only)"""
        losses = []
        for i, batch in enumerate(train_loader):
            
            batch = {key: tensor.to(device) for key, tensor in batch.items()}

            loss, _, _, _, _, _,_,_ = model(**batch)
            # loss, _, _, _, _ = model(**batch)

            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()   
        # if epoch in EXP_EPOCH:

        #     OUTPUTS_OF_ATTEN = [[] for i in range(12)]
        #     OUTPUTS_AFTER_ATTEN = [[] for i in range(12)]
        #     OUTPUTS_OF_FFN = [[] for i in range(12)]
        #     OUTPUTS_AFTER_FFN = [[] for i in range(12)]
        #     EMBEDDINGS = []
        #     for i, batch in enumerate(train_loader):

        #         if i%EXP_STEP == 0:
            
        #             batch = {key: tensor.to(device) for key, tensor in batch.items()}

        #             _, _, ATTOUTS, ATTOS, FFNOS, OUTSA = model(**batch)
        #             for a in len(ATTOUTS):
        #                 OUTPUTS_OF_ATTEN[a].append(ATTOUTS[a])
        #             for a in len(ATTOS):
        #                 OUTPUTS_AFTER_ATTEN[a].append(ATTOS[a])
        #             for a in len(FFNOS):
        #                 OUTPUTS_OF_FFN[a].append(FFNOS[a])
        #             for a in len(OUTSA[1:]):
        #                 OUTPUTS_AFTER_FFN[a].append(OUTSA[a+1])
                    
        #             EMBEDDINGS.append(OUTSA[0])
            
        
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator, device)
        # loss_valid2 = validate(model, val_loader2, accelerator, device)
        # accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Ahead Valid Loss: {loss_valid2}')
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}')


        if accelerator.is_local_main_process:
            writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar('perplexity_valid', loss_valid, epoch)
            # writer.add_scalar('ahead_perplexity_valid', loss_valid2, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    # accelerator.save_state('./bert-1103-stage0')
    torch.save(model,'1211-A.pth')
    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/bert-1213.json')
    # dataset = RestaurantForLM_small(config=config)
    dataset = Mixdata_1103(config = config)
    # ahead1 = RestaurantforLM_1103(config)
    # ahead2 = Mixdata_1103(config)
    # ahead_dataset = old_MixedData_after_stage1(config = config)
    
    device = torch.device("cuda")
    # model = base_models.BertForMLM_ELF_Z(config=config)
    # model = torch.load("1211-A.pth")
    model = base_models.BertForMLM_POST_C(config=config)
    # model = torch.load("/home/jxzhou/model/bert-pre-norm.bin")
    model.load_state_dict(torch.load("/home/jxzhou/model/bert-combine-residual-post-norm.bin"))
    # /home/jxzhou/model/bert-combine-residual-pre-norm.bin
    # /home/jxzhou/model/bert-post-norm.bin
    # /home/jxzhou/model/bert-pre-norm.bin
    # /home/jxzhou/model/bert-combine-residual-post-norm.bin
    model.to(device)
    # model = nn.DataParallel(model)
    
    # train(model=model, num_epochs=50, dataset=dataset, device=device)
    show_clustering(model,dataset,device,config)