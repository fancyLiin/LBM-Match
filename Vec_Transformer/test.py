import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import json
import torch.optim as optim
emblist1,emblist2,labellist = [],[],[]
temp = 0
# 假设embedding1和embedding2是两个包含embedding向量的列表
with open('../datasets_p/Testdatavec.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        qvec = data["qfact_vec"]
        cvec = data["cfact_vec"]
        label = data["label"]
        emblist1 = emblist1 + [qvec]
        emblist2 = emblist2 + [cvec]
        labellist = labellist + [int(label)]
        temp=temp+1
        if temp ==120:
            break
embedding1 = torch.tensor(emblist1[:100])
embedding2 = torch.tensor(emblist2[:100])
labels = torch.tensor(labellist[:100])  # 0表示一类，1表示另一类
Tembedding1 = torch.tensor(emblist1[-20:])
Tembedding2 = torch.tensor(emblist2[-20:])
Tlabels = torch.tensor(labellist[-20:])
print(labels)

class EmbeddingPairDataset(Dataset):
    def __init__(self, embedding1, embedding2, labels):
        self.embedding1 = embedding1
        self.embedding2 = embedding2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'embedding1': self.embedding1[idx],
            'embedding2': self.embedding2[idx],
            'labels': self.labels[idx]
        }

dataset = EmbeddingPairDataset(embedding1, embedding2, labels)
Tdataset = EmbeddingPairDataset(Tembedding1, Tembedding2, Tlabels)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(Tdataset, batch_size=2, shuffle=False)

class TransformerClassificationModel(nn.Module):
    def __init__(self, input_dim=1024, num_classes=4, num_layers=6, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(TransformerClassificationModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, embedding1, embedding2):
        x1 = self.embedding(embedding1).unsqueeze(1)
        x2 = self.embedding(embedding2).unsqueeze(1)
        x = torch.cat((x1, x2), dim=1)  # 合并两个嵌入向量
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # 对Transformer的输出进行池化
        x = self.classifier(x)
        return x

# 初始化模型
model = TransformerClassificationModel()

# 设置优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练模型
model.train()
for epoch in range(10):  # 训练10个epoch
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(batch['embedding1'], batch['embedding2'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch['embedding1'], batch['embedding2'])
        predictions = torch.argmax(outputs, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)
print(correct,total)
print(f'Accuracy: {correct / total}')