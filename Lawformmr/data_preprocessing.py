# 补充
import json
import re
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader


class DataPreprocessing:
    def __init__(self, tokenizer_name='bert-base-chinese', max_len=512):
        # 初始化BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def clean_text(self, text):
        """
        清洗文本：去除多余的空格、标点符号等
        """
        # 去除所有非中文字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        # 去除多余的空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_data(self, data_path):
        """
        处理原始数据，将其转化为BERT输入格式
        """
        texts = []
        labels = []
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                # 假设每条数据包含案件描述和标签
                query = data['q']  # 查询案件
                candidate = data['c']  # 候选案件
                label = int(data['label'])  # 标签

                # 清洗案件描述
                query = self.clean_text(query)
                candidate = self.clean_text(candidate)

                texts.append((query, candidate))
                labels.append(label)

        return texts, labels


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
        # 使用 tokenizer 对两个案件描述进行编码
        encoding = self.tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=True,  # 添加[CLS]和[SEP]特殊符号
            max_length=self.max_len,
            truncation=True,  # 超过最大长度的文本会被截断
            padding='max_length',  # 填充到最大长度
            return_attention_mask=True,
            return_tensors='pt'  # 返回PyTorch张量
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }


if __name__ == "__main__":
    # 设置文件路径和参数
    data_path = 'train_data.json'  # 训练数据文件路径
    tokenizer_name = 'bert-base-chinese'
    max_len = 512

    # 初始化数据预处理对象
    preprocessing = DataPreprocessing(tokenizer_name, max_len)

    # 处理数据
    texts, labels = preprocessing.process_data(data_path)

    # 创建数据集
    dataset = TextPairDataset(texts, labels, preprocessing.tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 示例：输出第一个批次数据
    for batch in data_loader:
        print(batch)
        break