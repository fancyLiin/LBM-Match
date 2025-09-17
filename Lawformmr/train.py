import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
import json
from lawformer import LawformerModel  # 假设已定义LawformerModel类
from data_preprocessing import DataPreprocessing, TextPairDataset  # 假设数据预处理模块

# 训练配置
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 512
LEARNING_RATE = 1e-5
TRAIN_DATA_PATH = 'train_data.json'
VAL_DATA_PATH = 'val_data.json'

# 初始化 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 初始化数据预处理类
preprocessing = DataPreprocessing(tokenizer_name='bert-base-chinese', max_len=MAX_LEN)

# 读取训练数据
train_texts, train_labels = preprocessing.process_data(TRAIN_DATA_PATH)
val_texts, val_labels = preprocessing.process_data(VAL_DATA_PATH)

# 创建数据集
train_dataset = TextPairDataset(train_texts, train_labels, tokenizer, max_len=MAX_LEN)
val_dataset = TextPairDataset(val_texts, val_labels, tokenizer, max_len=MAX_LEN)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型
model = LawformerModel(num_classes=4)  # 假设有4个类别
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()


# 训练函数
def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 模型前向传播
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


# 评估函数
def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 模型前向传播
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


# 训练并评估模型
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    # 训练
    avg_loss = train_model(model, train_loader, optimizer, criterion)
    print(f"Training Loss: {avg_loss}")

    # 评估
    accuracy = evaluate_model(model, val_loader)
    print(f"Validation Accuracy: {accuracy}")

# 保存模型
torch.save(model.state_dict(), 'lawformer_model.pth')
print("Model saved to 'lawformer_model.pth'")