import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.random_projection import GaussianRandomProjection
# Set a random seed for reproducibility
np.random.seed(42)

# Define centers for the four clusters
centers = [
    np.random.normal(0, 1, 768),
    np.random.normal(1, 1, 768),
    np.random.normal(-1, 1, 768),
    np.random.normal(2, 1, 768)
]

# Generate data around each center

# ACC = []
radius = 20
# for radius in range(100,5,-5):
data = {i:[] for i in range(4)}
for i, center in enumerate(centers):
    for _ in range(10000):
        # Add some noise to create a cluster around each center
        point = center + np.random.normal(0, radius/100, 768)
        data[i].append(point)
        # data.append({
        #     'label': f'class_{i+1}',
        #     'data': point
        # })
generated_data = []
labels = []
for j in range(4):
    generated_data+=data[j]
    labels+=[j]*10000
    # cn =5
ACC = []
for cn in range(2,21):
    transformer = GaussianRandomProjection(n_components=cn)
    reduced_data = transformer.fit_transform(generated_data)

    # plt.figure(cn)
    # for i in range(4):
    #     plt.scatter(reduced_data[i*10000:(i+1)*10000,0],reduced_data[i*10000:(i+1)*10000,1],label = i)
    # plt.legend()
    # plt.savefig("0702-test-%d.png"%radius)

    # 重新进行KMeans聚类
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(reduced_data)
    predicted_clusters = kmeans.predict(reduced_data)
    print(predicted_clusters)
    # 创建一个字典来存储每个簇和对应的标签计数
    cluster_label_counts = {i: {} for i in range(4)}

    # 对每个数据点，更新其簇的标签计数
    for cluster, label in zip(predicted_clusters, labels):
        cluster_label_counts[cluster].setdefault(label, 0)
        cluster_label_counts[cluster][label] += 1

    # 确定每个簇的主要标签
    cluster_labels = [max(labels_count, key=labels_count.get) for labels_count in cluster_label_counts.values()]
    # print(cluster_labels)

    # 将预测的簇标签转换为最终预测的标签
    predicted_labels = np.array([cluster_labels[cluster] for cluster in predicted_clusters])
    # print(predicted_labels)
    # 计算聚类的正确率
    accuracy = accuracy_score(labels, predicted_labels)
    ACC.append(accuracy)
    print(accuracy)
plt.figure(100)
plt.plot([cn for cn in range(2,21)],ACC)
plt.xlabel("Reduced Dimension")
plt.ylabel("Clustering Accuracy")
plt.savefig('0702-accuracy-dimension.png')