import json
from itertools import islice

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import tensor, nn
import torch.optim as optim


#
# class SimilarityDataset(Dataset):
#     def __init__(self, data_file, max_seq_length=None):
#         self.data = []
#         self.max_seq_length = max_seq_length
#
#         with open(data_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 item = json.loads(line.strip())  # 每行读取并解析为字典
#                 # 这里可以添加额外的验证或处理逻辑
#
#                 # 确保所有相似度列表的总长度不超过max_seq_length
#                 if self.max_seq_length and (
#                         len(item['cfact_cos_sim']) +
#                         len(item['qfact_cos_sim']) +
#                         len(item['charge_cos_sim'][0]) + 1  # 加上fact_cos_sim
#                 ) > self.max_seq_length:
#                     raise ValueError("A sample's combined similarity lists exceed the max_seq_length.")
#
#                 self.data.append(item)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         # 合并所有相似度值为一个序列
#         seq = [item['fact_cos_sim']] + item['cfact_cos_sim'] + item['qfact_cos_sim'] + item['charge_cos_sim'][0]
#
#         # 填充或截断序列到固定长度
#         seq = seq[:self.max_seq_length] + [0] * (self.max_seq_length - len(seq))  # 用0填充
#
#         label = item['label']
#         return tensor(seq, dtype=torch.float32), tensor(label, dtype=torch.long)
#
#
# # 示例用法
# dataset = SimilarityDataset('datacossim.json')


class SimilarityDataset(Dataset):
    def __init__(self, data_file, max_seq_length=float('inf')):
        self.data = []
        self.max_seq_length = max_seq_length if isinstance(max_seq_length, int) else 1

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())

                # 依次取元素并补0的处理函数
                def sequential_fill_with_zero(*lists):
                    valid_lists = [lst for lst in lists if isinstance(lst, list)]  # 过滤掉非列表项
                    if not valid_lists:  # 如果没有有效的列表，则返回空列表或根据需要处理
                        return []

                    max_len = max(len(lst) for lst in valid_lists)
                    result = []
                    for i in range(max_len):
                        for lst in valid_lists:
                            result.append(lst[i] if i < len(lst) else 0)
                    result = result + [0] * (self.max_seq_length + 1 - len(result))
                    return result[:self.max_seq_length]  # 限制长度

                # 处理并构造序列
                chargecos = sorted([item for sublist in item['charge_cos_sim'] for item in sublist], reverse=True)
                cfact_cos = sorted(item['cfact_cos_sim'], reverse=True)
                qfact_cos = sorted(item['qfact_cos_sim'], reverse=True)
                seq = [item['fact_cos_sim']]+sequential_fill_with_zero(
                    chargecos,
                    cfact_cos,
                    qfact_cos
                )

                self.data.append({
                    'seq': seq,
                    'label': int(item['label'])  # 确保标签转换为整数
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item['seq']
        label = item['label']

        # 序列已经在初始化时被正确填充到max_seq_length，此处无需再次填充
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return seq_tensor, label_tensor
# 示例用法
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
def train(tensorlist,labellist):
    numpy_array = np.array(tensorlist)
    tensor_data = torch.tensor(numpy_array, dtype=torch.float32).unsqueeze(1)
    labels_tensor = torch.tensor(labellist, dtype=torch.long)
    # 模型参数
    input_dim = 1
    hidden_dim = 64  # 隐藏层维度
    output_dim = 4  # 输出维度（对应0-3的类别）

    # 实例化模型、损失函数和优化器
    model = SimpleMLP(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()  # 适用于多分类问题的损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降优化器

    # 训练循环
    num_epochs = 1000
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(tensor_data)
        loss = criterion(outputs, labels_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:  # 每隔save_interval个epoch保存模型
            # torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')  # 保存模型状态字典
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Model saved.')

    # 训练循环结束后，保存最终模型
    torch.save(model.state_dict(), 'final_model2_10000.pth')
# 主函数
def main():
    tensorlist = []
    labellist = []
    # 打开文件
    with open('../Testdatacossim.json', 'r', encoding='utf-8') as file:
        # 逐行读取
        for line in file:                # 解析每行JSON
            json_obj = json.loads(line)
                # 提取label元素
            label = json_obj.get('label')
            labellist = labellist+[int(label)]

    with open('cos_tfidf_test.json', 'r', encoding='utf-8') as file:
        # 读取整个文件的内容
        json_data = file.read()
        tensorlist = json.loads(json_data)
        print(len(tensorlist))
    train(tensorlist[:3795],labellist[:3795])

if __name__ == '__main__':
    main()
