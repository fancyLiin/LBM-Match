import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, MambaForCausalLM, MambaConfig
from sklearn.model_selection import train_test_split
import json
import numpy as np

# 首先读取 legal.json 文件并将其内容存储在 json_content 变量中
with open('legal.json', 'r', encoding='utf-8') as file:
    json_content = json.load(file)


# 定义您的数据集类
class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"A案例: {item['query']} B候选文本: {item['candidate']} B罪名: {str([json_content[id]['context'] for id in item['charge'] if id in json_content])} B涉及的法条: {str(item['article'])}"
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 添加特殊标记
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor([int(item['label'])], dtype=torch.long)
        }


# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

# 读取JSON数据集
with open('train.json', 'r', encoding='utf-8') as file:
    dataset = [json.loads(line) for line in file]

# 划分训练集和验证集
train_data, val_data = train_test_split(dataset, test_size=0.2)

# 创建数据加载器
train_dataset = MyDataset(train_data, tokenizer, max_length=4096)
val_dataset = MyDataset(val_data, tokenizer, max_length=4096)


# 定义一个collate函数，用于将数据打包成批次
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# 加载模型
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

# 定义损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 3  # 训练轮数

for epoch in range(num_epochs):
    print('开始训练')
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证模型
    model.eval()
    val_loss = 0.0
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

# 保存模型
model.save_pretrained("save/model")