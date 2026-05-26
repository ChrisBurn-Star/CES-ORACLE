import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from scipy.interpolate import UnivariateSpline
# 从文件中读取数据
with open("/home/jxzhou/PLM_PER/BERT-CL-main/0625-huawei.txt", "r") as f:
    lines = f.readlines()
CENTERS = []
RADIUS = []
RADIO = []
for i in range(10,46871,10):
    CENTERS.append(torch.load('MOE_FOR_HUAWEI_PTH/CENTERS-ATT-steps%d.pth'%i))
    RADIUS.append(torch.load('MOE_FOR_HUAWEI_PTH/RADIUS-ATT-steps%d.pth'%i))
    RADIO.append(torch.load('MOE_FOR_HUAWEI_PTH/CLUSTER-IDS-ATT-steps%d.pth'%i))
# layer = 11
# 提取第 11 层的数据
for layer in range(12):
    layer_11_data = []
    for line in lines:
        if "layer %3d    "%layer in line:
            layer_11_data.append(line)

    # 解析数据
    layer_11_data = [eval(line.split("layer %3d    "%layer)[1]) for line in layer_11_data]
    print(layer_11_data)
    plt.figure(layer)
    # 绘制图表 5
    for i in layer_11_data[0]:
        print(len(layer_11_data))
        spl = UnivariateSpline(np.arange(0, 46870, 20), [layer_11_data[j][i] for j in range(0,len(layer_11_data),2)], k=10)

        # 生成更多的x值用于平滑曲线
        x_new = np.arange(0, 46870, 10)
        y_smooth = spl(x_new)
        # plt.plot(np.arange(0, 46870, 20), [layer_11_data[j][i] for j in range(0,len(layer_11_data),2)],linewidth=0.7,ls="-",marker=',',label = i)
        plt.plot(x_new,y_smooth,linewidth=0.7,ls="-",marker=',',label = i)

        # 设置图表标题和坐标轴标签
        plt.title("Layer %d Workload"%layer,fontsize=20)
        plt.xlabel("Steps",fontsize=20)
        plt.ylabel("Workload",fontsize=20)
        # plt.yticks(np.arange(0, 0.5, 0.02))
        # plt.xticks(np.arange(0, 46870, 5))
        plt.legend()
    plt.savefig('0625-moe-huawei-workload-layer%d.png'%layer)
# COLOR = ['r','g','b','y','pink','m','c','gray']

# CENTERS_ = []

# for k in range(len(CENTERS)):
#     for layer in range(12):
#         for i in range(len(CENTERS[k][layer])):
#             CENTERS_.append(CENTERS[k][layer][i])
# # print(len(CENTERS_))
# CENTERS_ = np.array(CENTERS_)
# # print(CENTERS_.shape)
# UU = umap.UMAP(random_state=42).fit(CENTERS_)


# for k in range(len(CENTERS)):
#     for layer in range(12):
#         for i in range(len(CENTERS[k][layer])):
#             # print(RADIUS[k][layer][i])
#             print(k,layer,i)
#             plt.Circle(UU.transform(CENTERS[k][layer][i]), RADIUS[k][layer][i], label = i, fill=False)

#             plt.legend()
#         plt.savefig('0625-moe-huawei/0623-moe-huawei-clustering-layer%d-steps%d.png'%(layer,k))

CENTERS_ = [{i:[] for i in range(8)}for l in range(12)]
RADIUS_ = [{i:[] for i in range(8)}for l in range(12)]
RADIO_ = [{i:[] for i in range(8)}for l in range(12)]
for k in range(0,len(CENTERS),20):

    for layer in range(12):
        s = 0
        for i in range(len(CENTERS[k][layer])):
            s+=RADIO[k][layer][i]
        print(s)
        for i in range(len(CENTERS[k][layer])):
            CENTERS_[layer][i].append(CENTERS[k][layer][i].mean())
            RADIUS_[layer][i].append(RADIUS[k][layer][i]/1000)
            RADIO_[layer][i].append(RADIO[k][layer][i]/s)
    
for layer in range(12):
    plt.figure(layer*10000)
    for i in range(len(CENTERS_[layer])):
        print(layer,i)
        spl = UnivariateSpline(np.arange(0,len(CENTERS),20), RADIO_[layer][i], k=5)

        # 生成更多的x值用于平滑曲线
        x_new = np.arange(0,len(CENTERS),10)
        y_smooth = spl(x_new)

        plt.title("Layer %d CLuster Workload"%layer,fontsize=20)
        plt.xlabel("Steps",fontsize=20)
        plt.ylabel("Workload",fontsize=20)

        # print(RADIUS_[i])
        # plt.errorbar([i for i in range(len(CENTERS_[layer][i]))], CENTERS_[layer][i], yerr=RADIUS_[layer][i], fmt='-o', ecolor='red', capsize=5)
        # plt.plot(np.arange(0,len(CENTERS),20), RADIO_[layer][i],linewidth=0.7,ls="-",marker=',',label = i)
        plt.plot(x_new*10, y_smooth,linewidth=0.7,ls="-",marker=',',label = i)
        
        # plt.fill_between([i for i in range(len(CENTERS_[layer][i]))],np.array(CENTERS_[layer][i])-np.array(RADIUS_[layer][i]),np.array(CENTERS_[layer][i])+np.array(RADIUS_[layer][i]), label = i)
    
        plt.legend()
    plt.savefig('0625-moe-huawei/0625-moe-huawei-clustering-layer%d.png'%(layer))