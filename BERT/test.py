import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import json

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('path_to_your_saved_model.pth', num_labels=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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


# 读取验证集数据
val_texts = []
val_labels = []
with open('TR_test.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        text_pair = (data["q"], data["c"])
        val_texts.append(text_pair)
        val_labels.append(int(data["label"]))

# 创建数据加载器
val_dataset = TextPairDataset(val_texts, val_labels, tokenizer, max_len=512)
val_loader = DataLoader(val_dataset, batch_size=16)

# 收集所有预测和真实标签
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, 1)
        probs = torch.softmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy}")

# 将标签和概率转换为NumPy数组
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# 计算AUC-ROC (需要one-hot编码的标签和概率)
auc_roc = roc_auc_score(np.eye(4)[all_labels], all_probs, multi_class="ovo")
print(f"AUC-ROC: {auc_roc}")

# 计算PR-AUC
pr_auc = average_precision_score(np.eye(4)[all_labels], all_probs, average="macro")
print(f"Mean PR-AUC: {pr_auc}")
#
# Epoch: 1, Loss: 0.9102597924818595
# Accuracy: 0.681126173096976
# AUC-ROC: 0.6798537599478566
# Mean PR-AUC: 0.3658889397236488
# Epoch: 2, Loss: 0.969551960589985
# Accuracy: 0.6767466110531803
# AUC-ROC: 0.6844643939396051
# Mean PR-AUC: 0.36666793718948354
