# 补充
# data_preprocessing.py
import json
import re
from transformers import AutoTokenizer


class DataPreprocessing:
    def __init__(self, tokenizer_name='ChatGLM3', max_len=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def clean_text(self, text):
        """
        清洗文本：去除多余的空格、标点符号等
        """
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)  # 去除所有非中文字符
        text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
        return text

    def process_data(self, data_path):
        """
        处理原始数据，将其转化为ChatGLM3输入格式
        """
        texts = []
        labels = []
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                query = data['q']  # 查询案件
                candidate = data['c']  # 候选案件
                label = int(data['label'])  # 标签
                query = self.clean_text(query)
                candidate = self.clean_text(candidate)
                texts.append((query, candidate))
                labels.append(label)

        return texts, labels