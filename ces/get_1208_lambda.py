import numpy as np
import matplotlib.pyplot as plt
import csv
# 假设你的张量是 tensor，将其转换为 NumPy 数组

read_tensors = []
with open('my_tuple.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        # 读取每一行，并将数据恢复成张量形状
        tensor_data = np.array(row, dtype=float)  # 假设数据是浮点数类型
        # tensor_shape = (2, 2)  # 恢复张量的形状，这里假设是2x2的张量
        tensor = tensor_data
        read_tensors.append(tensor)
print(len(read_tensors))
# 将张量转换为 NumPy 数组
# # tensor_np = np.array(tensor)

# # 计算特征值
# # 注意：这里使用 np.linalg.eigvals() 函数计算特征值
# # eig_vals 是一个包含特征值的数组
# eig_vals = np.linalg.eigvals(tensor_np.T @ tensor_np)

# # 绘制特征值图
# dimensions = len(eig_vals)
# x = np.arange(1, dimensions + 1)  # 横坐标为维度d
# y = eig_vals  # 纵坐标为特征值

# plt.figure(figsize=(8, 6))
# plt.scatter(x, y)
# plt.title('Eigenvalues')
# plt.xlabel('Dimension')
# plt.ylabel('Eigenvalue')
# plt.grid(True)
# plt.show()