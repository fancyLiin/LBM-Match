# evaluate.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
from model import LawformerModel  # 假设LawformerModel在model目录下
from data_preprocessing import TextPairDataset  # 假设数据预处理模块

# 配置
BATCH_SIZE = 16
MAX_LEN = 512
MODEL_PATH = 'fine_tuned_ChatGLM3'  # 微调后模型路径
VAL_DATA_PATH = 'val_data.json'  # 验证集路径

# 初始化 BERT Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# 加载验证数据
def read_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts.append((data['q'], data['c']))  # 查询和候选案件描述
            labels.append(int(data['label']))  # 标签
    return texts, labels


val_texts, val_labels = read_data(VAL_DATA_PATH)

# 创建验证集数据集
val_dataset = TextPairDataset(val_texts, val_labels, tokenizer, max_len=MAX_LEN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 加载训练好的模型
model = LawformerModel(num_classes=4)  # 假设有4个类别
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))  # 加载模型权重
model.eval()


# 评估模型
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


# 评估并输出结果
accuracy = evaluate_model(model, val_loader)
print(f"Validation Accuracy: {accuracy:.4f}")