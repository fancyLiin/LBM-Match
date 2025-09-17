# train.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_preprocessing import DataPreprocessing  # 数据预处理

# 配置
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 512
LEARNING_RATE = 1e-5
TRAIN_DATA_PATH = 'train_data.json'
VAL_DATA_PATH = 'val_data.json'

# 初始化 Llama 2 模型和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained('Llama-2')
model = AutoModelForCausalLM.from_pretrained('Llama-2')

# LoRA 配置
lora_config = LoraConfig(
    r=8,  # LoRA中的低秩矩阵的秩
    lora_alpha=16,  # LoRA中的缩放系数
    lora_dropout=0.1,  # Dropout率
    bias="none",  # 在LoRA层中的偏置设置
)

# 准备模型进行LoRA微调
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 数据预处理
preprocessing = DataPreprocessing(tokenizer_name='Llama-2', max_len=MAX_LEN)
train_texts, train_labels = preprocessing.process_data(TRAIN_DATA_PATH)
val_texts, val_labels = preprocessing.process_data(VAL_DATA_PATH)

# 创建数据集
train_dataset = TextPairDataset(train_texts, train_labels, tokenizer, max_len=MAX_LEN)
val_dataset = TextPairDataset(val_texts, val_labels, tokenizer, max_len=MAX_LEN)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 设置优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# 训练模型
def train_model(model, train_loader, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}")
        for batch in loop:
            inputs = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            optimizer.zero_grad()

            # 模型前向传播
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} finished. Avg loss: {avg_loss}")


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


# 训练并评估模型
for epoch in range(3):  # 假设训练3个epoch
    train_model(model, train_loader, optimizer, num_epochs=1)
    accuracy = evaluate_model(model, val_loader)
    print(f"Validation Accuracy: {accuracy:.4f}")

# 保存微调后的模型
model.save_pretrained("fine_tuned_Llama2")
tokenizer.save_pretrained("fine_tuned_Llama2")