import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
import json
# 定义数据集类
class TextPairDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text1, text2 = self.texts[item]
        # 使用encode_plus方法，设置max_length和truncation_strategy
        encoding = self.tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation='longest_first',
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }
# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
train_texts = []
test_texts = []
train_labels = []
test_labels = []
with open('TR_train.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        text_cp =(data["q"],data["c"])
        train_texts = train_texts +[text_cp]
        train_labels = train_labels +[int(data["label"])]
with open('TR_test.json', 'r', encoding='utf-8') as f1:
    for line in f1:
        data = json.loads(line)
        text_cp =(data["q"],data["c"])
        test_texts = test_texts +[text_cp]
        test_labels = test_labels +[int(data["label"])]


# 创建数据加载器
train_dataset = TextPairDataset(train_texts, train_labels, tokenizer, max_len=128)
test_dataset = TextPairDataset(test_texts, test_labels, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=1e-5)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
print("开始训练")
for epoch in range(num_epochs):
    print(f"开始第{epoch + 1} 轮训练")
    model.train()
    print("开始一个train_batch")
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 评估模型
    model.eval()
    total_loss = 0
    print("开始一个test_batch")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    print(f"Epoch: {epoch+1}, Loss: {total_loss/len(test_loader)}")