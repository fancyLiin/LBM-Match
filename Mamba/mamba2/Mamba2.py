import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from torch.utils.data import DataLoader, TensorDataset
from mamba_ssm import Mamba2
import json
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, roc_curve, auc
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar
from mamba_ssm.ops.triton.layer_norm import RMSNorm

batch_size = 32
batch_len = int(19169 / batch_size)
batch = []
num_epochs = 10

Temblist1,Temblist2,Tlabellist = [],[],[]
with open('autodl-tmp/Testdatavec.json', 'r') as f:
    for line in f:
        Tdata = json.loads(line)
        Tqvec = Tdata["qfact_vec"]
        Tcvec = Tdata["cfact_vec"]
        Tlabel = Tdata["label"]
        Temblist1 = Temblist1 + [Tqvec]
        Temblist2 = Temblist2 + [Tcvec]
        Tlabellist = Tlabellist + [int(Tlabel)]
# Sample data preparation (replace this with your actual data)
# Assuming embeddings1 and embeddings2 are tensors of shape (num_samples, 1024)
# and labels is a tensor of shape (num_samples,)
Tembedding1 = torch.tensor(Temblist1,device="cuda")
Tembedding2 = torch.tensor(Temblist2,device="cuda")
Tlabels = torch.tensor(Tlabellist,device="cuda")
# Concatenate the embeddings
combined_embeddings = torch.cat((Tembedding1, Tembedding2), dim=1)
combined_embeddings = combined_embeddings.unsqueeze(1)
# Create a DataLoader
dataset = TensorDataset(combined_embeddings, Tlabels)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
print("读取完成")

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class CombinedModel(nn.Module):
    def __init__(self, num_layers, d_model, d_state, d_conv, expand, num_heads=8, d_ff=4096, num_transformer_layers=0):
        super(CombinedModel, self).__init__()
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff)
            for _ in range(num_transformer_layers)
        ])
        self.mamba_layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': RMSNorm(d_model),
                'mixer': Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

            }) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, 4)  # 分类任务有4个类别

    def forward(self, x):
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)

        for i, layer in enumerate(self.mamba_layers):
            x = layer['mixer'](x)
            x = layer['norm'](x)

        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


# Initialize the base model
num_layers = 1  # 设置为6层
dim = 3072*4  # d_model 设置为 1024 以匹配连接后的嵌入向量
base_model = CombinedModel(num_layers=num_layers, d_model=dim, d_state=64, d_conv=4, expand=2).to("cuda")
# Define loss and optimizer
temp = 0
for epoch in range(num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.parameters(), lr=0.001)
    base_model.train()
    running_loss = 0.0
    combined_tensor_list, labellist = [], []
    bar = Bar('Epoch' + str(epoch + 1), max=batch_len)
    with open('autodl-tmp/GPTvec/Trian_ChatGPT_datavec.json', 'r') as f:
        for line in f:
            temp = temp + 1
            data = json.loads(line)
            batch.append(data)
            qvec = data["qfact_vec"]
            cvec = data["cfact_vec"]
            label = data["label"]
            qvec1 = data["qcharge_vec"]
            cvec1 = data["ccharge_vec"]
            labellist = labellist + [int(label)]
            Ts_qvec = torch.tensor(qvec, device="cuda")
            Ts_cvec = torch.tensor(cvec, device="cuda")
            Ts_qvec1 = torch.tensor(qvec1, device="cuda") #.unsqueeze(0)
            Ts_cvec1 = torch.tensor(cvec1, device="cuda")#.unsqueeze(0)
            # seg_vector = torch.ones(1024, device="cuda").unsqueeze(0)
            combined_tensor = torch.cat(
                (Ts_qvec,Ts_cvec, Ts_qvec1, Ts_cvec1), dim=0)
            print(combined_tensor.size)
            # x = combined_tensor.size(0)
            # padding_rows = 128 - x
            # padding = torch.zeros(padding_rows, 1024, device="cuda")
            # 堆叠张量
            # padded_tensor = torch.cat((combined_tensor, padding), dim=0)
            combined_tensor_list = combined_tensor_list + [combined_tensor]
            if temp % batch_size == 0:
                # 在这里处理批次数据，例如将其转换为张量并传递给模型
                stacked_tensor = torch.stack(combined_tensor_list)
                labels = torch.tensor(labellist, device="cuda")
                optimizer.zero_grad()
                outputs = base_model(stacked_tensor)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # 进行梯度裁剪
                # utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
                running_loss += loss.item()
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / batch_len}")
                combined_tensor_list, labellist = [], []
                bar.next()
        bar.finish()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / batch_len}")
        torch.save(base_model.state_dict(), f"autodl-tmp/mamba2_model_epoch_{epoch + 1}.pth")
        print(f"Model saved after epoch {epoch + 1}.")
    base_model.eval()
    all_preds = []
    all_labels = []
    all_preds_prob = []
    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            batch_embeddings, batch_labels = batch_embeddings.to("cuda"), batch_labels.to("cuda")
            outputs = base_model(batch_embeddings).squeeze(1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            preds_prob = torch.softmax(outputs, dim=1)
            all_preds_prob.extend(preds_prob.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy}")

    # Convert lists to numpy arrays
    all_preds_prob = np.array(all_preds_prob)
    all_labels = np.array(all_labels)

    # Calculate mean AUC-ROC and mean PR-AUC
    all_labels_one_hot = nn.functional.one_hot(torch.tensor(all_labels), num_classes=4).numpy()

    mean_auc_roc = roc_auc_score(all_labels_one_hot, all_preds_prob, average='macro', multi_class='ovo')
    mean_pr_auc = average_precision_score(all_labels_one_hot, all_preds_prob, average='macro')

    print(f"Mean AUC-ROC: {mean_auc_roc}")
    print(f"Mean PR-AUC: {mean_pr_auc}")
    # # Plotting ROC curves for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()

    # for i in range(4):
    #     fpr[i], tpr[i], _ = roc_curve(all_labels_one_hot[:, i], all_preds_prob[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    # # Plot ROC curves
    # plt.figure()
    # for i in range(4):
    #     plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves for Each Class')
    # plt.legend(loc='lower right')
    # plt.savefig('roc_curves.png')
    # plt.show()
    # print("ROC curves saved.")