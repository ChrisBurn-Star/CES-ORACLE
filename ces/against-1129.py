import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm


# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 加载MNIST数据集
trainset = torchvision.datasets.MNIST('/home/jxzhou/PLM_PER/MNIST', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST('/home/jxzhou/PLM_PER/MNIST', train=False, download=True, transform=transform)

# 按类别取出每个类别各1000条数据作为10个子数据集
class_datasets = {}
count_per_class = {i: 0 for i in range(10)}
for idx, (image, label) in enumerate(trainset):
    if count_per_class[label] < 1000:
        if label not in class_datasets:
            class_datasets[label] = []
        class_datasets[label].append((image, label))
        count_per_class[label] += 1
    if all(count == 1000 for count in count_per_class.values()):
        break

# 定义全连接神经网络模型
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = Flatten()
        self.fc1 = nn.Linear(784, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        im_rp = x
        x = self.fc2(x)
        return x, im_rp

# 定义模型训练函数
def train_model(trainloader,model, aheadloader):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(3):  # 假设每个类别进行3个epoch的训练
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs,_ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('[Epoch %d, Batch %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
    # running_loss = 0.0
    # for i, data in enumerate(aheadloader, 0):
    #     inputs, labels = data
    #     optimizer.zero_grad()

    #     outputs,_ = model(inputs)
    #     loss = criterion(outputs, labels)
    #     running_loss += loss.item()
    # print(f'ahead 0 class loss:{running_loss/(i+1)}')


    return model
def pca(input, threshold=0.80):
    # X = input.mean(axis=1)
    X = input.cpu().numpy()
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    # pca = PCA(n_components=X.shape[1])
    
    # explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
    # num_components = np.argmax(explained_variance_ratio >= threshold) + 1

    pca = PCA(n_components=2)
    X_pca_efficient = pca.fit_transform(X_std)    
    X_pca_efficient = torch.tensor(X_pca_efficient) 


    
    return X_pca_efficient

def show_reps(trainloader,model):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(1):  # 假设每个类别进行3个epoch的训练
        running_loss = 0.0
        REPS = []
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data
            

            outputs,res = model(inputs)
            REPS.append(res)
    # REPS = torch.tensor(REPS)
    REPS = torch.cat(REPS)
    # print(REPS.shape)

    PCAED_REPS = pca(REPS.detach())
    plt.figure(1)
    print(PCAED_REPS.shape)
    colors = plt.get_cmap('tab10', 10)
    for i in range(10):
        plt.scatter(PCAED_REPS[1000*i:1000*(i+1),0], PCAED_REPS[1000*i:1000*(i+1),1],color=colors(i),label=str(i),alpha=0.2)
    plt.legend()
    
    
    plt.savefig('1129-1.png')

    return 0







def show_overlop(model,set_old,set_new,o,n):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    ACTIVED_PARAS1 = []
    ACTIVED_PARAS2 = []
    ACTIVED_TENSORS1 = []
    ACTIVED_TENSORS2 = []
    lrs = 1e-5
    for epoch in range(1):
        c=0  # 假设每个类别进行3个epoch的训练
        for i, data in enumerate(set_old, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs,_ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            c+=1
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.all(param.grad == 0):
                        ACTIVED_PARAS1.append(0)
                        ACTIVED_TENSORS1.append(0)
                        pass
                    else:
                        ACTIVED_PARAS1.append(name)
                        ACP = torch.where(torch.abs(param.grad) > lrs, torch.tensor(1), param.grad)
                        ACP = torch.where(torch.abs(ACP) <=lrs , torch.tensor(-1), ACP)
                        ACTIVED_TENSORS1.append(ACP)
                else:
                    ACTIVED_PARAS1.append(0)
                    ACTIVED_TENSORS1.append(0)
            optimizer.zero_grad()
    for epoch in range(1):
        c=0  # 假设每个类别进行3个epoch的训练
        for i, data in enumerate(set_new, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs,_ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            c+=1
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.all(param.grad == 0):
                        ACTIVED_PARAS2.append(0)
                        ACTIVED_TENSORS2.append(0)
                        pass
                    else:
                        ACTIVED_PARAS2.append(name)
                        ACP = torch.where(torch.abs(param.grad) > lrs, torch.tensor(1), param.grad)
                        ACP = torch.where(torch.abs(ACP) <=lrs , torch.tensor(0), ACP)
                        ACTIVED_TENSORS2.append(ACP)
                else:
                    ACTIVED_PARAS2.append(0)
                    ACTIVED_TENSORS2.append(0)
            optimizer.zero_grad()
    repeat_count = 0
    all_count = 0

    total_params = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad)

    for l in range(len(ACTIVED_PARAS2)):
        para_name = ACTIVED_PARAS2[l]
        if para_name == ACTIVED_PARAS1[l] and para_name != 0:
            all_count+=ACTIVED_TENSORS1[l].numel()
            TEMP = ACTIVED_TENSORS1[l] - ACTIVED_TENSORS2[l]
            repeat_count+= torch.sum(torch.eq(TEMP, 0))

            
    print(repeat_count)
    overlop = repeat_count/(total_params*c)
    # print(f'old:{o} new:{n} {overlop}')
    print(overlop.item())



    return overlop


# 将10个类别的数据分别取1000条，依次训练模型


model = SimpleNN()
# for class_idx in range(10):
#     class_data = class_datasets[class_idx]
#     class_loader = torch.utils.data.DataLoader(class_data, batch_size=64, shuffle=True)

#     class_data0 = class_datasets[0]
#     aheadloader = torch.utils.data.DataLoader(class_data0, batch_size=64, shuffle=True)

    
#     # print(f"Training model for class {class_idx}...")
#     model = train_model(class_loader, model, aheadloader)
    # for nnl in range(class_idx):
    #     class_data_old = class_datasets[nnl]
    #     class_loader_old = torch.utils.data.DataLoader(class_data_old, batch_size=64, shuffle=True)
    #     overlop = show_overlop(model,class_loader_old, class_loader,nnl,class_idx)
# 将这10个类别的数据各取1000条混合成一个总的数据集，训练模型


mixed_dataset = []
for label in range(10):
    mixed_dataset.extend(class_datasets[label])


mixed_dataset_show = []

mixed_dataset_show.extend(class_datasets[0])
mixed_dataset_show.extend(class_datasets[1])
mixed_dataset_show.extend(class_datasets[2])



# mixed_dataset_show.extend(class_datasets[1])
# mixed_dataset_show.extend(class_datasets[4])
# mixed_dataset_show.extend(class_datasets[9])


# mixed_dataset_show.extend(class_datasets[5])
# mixed_dataset_show.extend(class_datasets[6])
# mixed_dataset_show.extend(class_datasets[7])





mixed_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=64, shuffle=True)

mixed_loader_show = torch.utils.data.DataLoader(mixed_dataset_show, batch_size=64, shuffle=False)
mixed_loader_show_all = torch.utils.data.DataLoader(mixed_dataset, batch_size=64, shuffle=False)
class_data0 = class_datasets[0]
class_loader_ahead = torch.utils.data.DataLoader(class_data0, batch_size=64, shuffle=True)

# p = show_reps(mixed_loader_show_all,model)
# for i in range(10):
#     class_data_old = class_datasets[i]
#     class_loader_old = torch.utils.data.DataLoader(class_data_old, batch_size=64, shuffle=True)
#     for j in range(10):
#         class_data_new = class_datasets[j]
#         class_loader_new = torch.utils.data.DataLoader(class_data_new, batch_size=64, shuffle=True)
#         overlop = show_overlop(model,class_loader_old, class_loader_new,i,j)

# torch.save(model, '1129-MNIST-INCREAMENTAL.pth')



print("Training model on mixed dataset...")
mixed_model = train_model(mixed_loader,model,class_loader_ahead)
p = show_reps(mixed_loader_show_all,model)

# for i in range(10):
#     class_data_old = class_datasets[i]
#     class_loader_old = torch.utils.data.DataLoader(class_data_old, batch_size=64, shuffle=True)
#     for j in range(10):
#         class_data_new = class_datasets[j]
#         class_loader_new = torch.utils.data.DataLoader(class_data_new, batch_size=64, shuffle=True)
#         overlop = show_overlop(mixed_model,class_loader_old, class_loader_new,i,j)
