import json
from itertools import islice
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import tensor, nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
    tensor_data = torch.tensor(numpy_array, dtype=torch.float32)
    labels_tensor = torch.tensor(labellist, dtype=torch.long)
    # 模型参数
    input_dim = 2
    hidden_dim = 64  # 隐藏层维度
    output_dim = 4  # 输出维度（对应0-3的类别）

    # 实例化模型、损失函数和优化器
    model = SimpleMLP(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()  # 适用于多分类问题的损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降优化器

    # 训练循环
    num_epochs = 10000
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(tensor_data)
        loss = criterion(outputs, labels_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model, 'entire_model.pth')
def eval(tensorlist,true_labels):
    numpy_array = np.array(tensorlist)
    tensor_data = torch.tensor(numpy_array, dtype=torch.float32)
    # labels_tensor = torch.tensor(true_labels, dtype=torch.long)
    model = SimpleMLP(2, 64, 4)
    # 加载训练好的权重
    model.load_state_dict(torch.load('final_model2_100000.pth'))
    # model = torch.load('final_model.pth')
    model.eval()
    with torch.no_grad():  # 禁用梯度计算以提高效率
        outputs = model(tensor_data)
        # 获取模型的原始输出（logits）
        logits = outputs

        # 将logits转换为概率
        probabilities = torch.softmax(logits, dim=1)

        # 获取每个样本的预测类别
        _, predicted_classes = torch.max(outputs, 1)
    probabilities_np = probabilities.detach().numpy()
    predicted_classes_np = predicted_classes.numpy()

    num_classes = probabilities_np.shape[1]
    roc_aucs = []
    pr_aucs = []

    for i in range(num_classes):
        # 将当前类标记为1，其余类标记为0，形成一对多的二分类问题
        true_labels = np.array(true_labels)

        # 然后进行后续操作
        binary_true_labels = (true_labels == i).astype(int)
        # 计算AUC-ROC
        roc_auc = roc_auc_score(binary_true_labels, probabilities_np[:, i])
        roc_aucs.append(roc_auc)

        # 计算PR-AUC
        pr_auc = average_precision_score(binary_true_labels, probabilities_np[:, i])
        pr_aucs.append(pr_auc)

    # 计算平均AUC-ROC和PR-AUC
    mean_roc_auc = np.mean(roc_aucs)
    mean_pr_auc = np.mean(pr_aucs)

    print(f"Mean AUC-ROC: {mean_roc_auc}")
    print(f"Mean PR-AUC: {mean_pr_auc}")
    # 计算准确率
    accuracy = accuracy_score(true_labels, predicted_classes)
    print(f"Accuracy: {accuracy}")

    # 计算召回率
    recall = recall_score(true_labels, predicted_classes, average='micro')  # 'macro'平均召回率，适用于多分类
    print(f"Recall: {recall}")

    # 计算F1分数
    f1 = f1_score(true_labels, predicted_classes, average='micro')  # 'macro'平均F1分数
    print(f"F1 Score: {f1}")
    if not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)

    num_classes = probabilities_np.shape[1]
    roc_aucs = []
    pr_aucs = []

    for i in range(num_classes):
        binary_true_labels = (true_labels == i)

        if binary_true_labels.any():
            fpr, tpr, _ = roc_curve(binary_true_labels, probabilities_np[:, i])
            roc_auc = roc_auc_score(binary_true_labels, probabilities_np[:, i])
            roc_aucs.append(roc_auc)

            pr_auc = average_precision_score(binary_true_labels, probabilities_np[:, i])
            pr_aucs.append(pr_auc)

            # 绘制第 i 类的 ROC 曲线
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')  # 绘制随机分类器的对角线
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic for class {i}')
            plt.legend(loc="lower right")
            plt.show()
        else:
            print(f"No instances of class {i} found.")
# 主函数
def main():
    dataset = SimilarityDataset('../Testdatacossim.json')  # 假设最大序列长度为10
    tensorlist = []
    labellist = []
    for i in range(min(len(dataset), len(dataset))):  # 查看前5个样本
        seq, label = dataset[i]
        # print(f"Sample {i+1}:")
        tensorlist = tensorlist + [seq.tolist()]
        labellist = labellist + [label.item()]
        # print(f"Sequence: {seq}, Label: {label.item()}")  # 使用tolist()将tensor转为list以便打印
    print(len(tensorlist))
    eval(tensorlist,labellist)
    # train(tensorlist,labellist)

if __name__ == '__main__':
    main()
# Mean AUC-ROC: 0.6019867205665106
# Mean PR-AUC: 0.29904989609015564
# Accuracy: 0.6736183524504692
# Recall: 0.6736183524504692
# F1 Score: 0.6736183524504692

# Mean AUC-ROC: 0.6440552300619529
# Mean PR-AUC: 0.3188114449667451
# Accuracy: 0.6736183524504692
# Recall: 0.6736183524504692
# F1 Score: 0.6736183524504692   7
#
# Mean AUC-ROC: 0.6642361874565355
# Mean PR-AUC: 0.3381148479706368
# Accuracy: 0.6771637122002085
# Recall: 0.6771637122002085
# F1 Score: 0.6771637122002085   4

# Mean AUC-ROC: 0.5862349373423565
# Mean PR-AUC: 0.30151812215528767
# Accuracy: 0.6736183524504692
# Recall: 0.6736183524504692
# F1 Score: 0.6736183524504692   1

# Mean AUC-ROC: 0.6759575543306137
# Mean PR-AUC: 0.3496150639792003
# Accuracy: 0.6782064650677789
# Recall: 0.6782064650677789
# F1 Score: 0.6782064650677789


# Mean AUC-ROC: 0.699061870385741
# Mean PR-AUC: 0.36038273016657896
# Accuracy: 0.670281543274244
# Recall: 0.670281543274244
# F1 Score: 0.670281543274244