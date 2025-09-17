import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# 加载数据集
class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index][0]
        label = self.data[index][1]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 参数设置
epochs = 3
max_len = 128
batch_size = 16
learning_rate = 2e-5

# 加载预训练模型和BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
model.to(device)

# 准备数据
# 这里需要你根据实际情况准备数据，data应该是一个列表，每个元素是一个元组，元组第一个元素是文本，第二个元素是标签
data = [("这是第一条数据", 0), ("这是第二条数据", 1), ("这是第三条数据", 2), ("这是第四条数据", 3)]

# 划分数据集
train_size = int(0.8 * len(data))
train_dataset = MyDataset(data[:train_size], tokenizer, max_len)
val_dataset = MyDataset(data[train_size:], tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练模型
model.train()
for epoch in range(epochs):
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# 评估模型
model.eval()
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        predictions = torch.argmax(logits, dim=-1)
        print(predictions)
