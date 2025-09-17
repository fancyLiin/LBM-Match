# 补充
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from transformers import BertTokenizer
import json
from lawformer import LawformerModel  # 假设已经定义了LawformerModel类
from data_preprocessing import TextPairDataset  # 假设数据预处理模块

# 配置
BATCH_SIZE = 16
MAX_LEN = 512
MODEL_PATH = 'lawformer_model.pth'  # 训练后保存的模型路径
VAL_DATA_PATH = 'val_data.json'  # 验证集路径

# 初始化 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


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
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 模型推理
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    auc_roc = roc_auc_score(np.eye(4)[all_labels], all_probs, multi_class="ovo")
    pr_auc = average_precision_score(np.eye(4)[all_labels], all_probs, average="macro")

    return accuracy, auc_roc, pr_auc


# 评估并输出结果
accuracy, auc_roc, pr_auc = evaluate_model(model, val_loader)

print(f"Accuracy: {accuracy}")
print(f"AUC-ROC: {auc_roc}")
print(f"Mean PR-AUC: {pr_auc}")