from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 初始化BERT模型和分词器
model_name = 'bert-base-chinese'  # 使用中文预训练模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 将模型设置为评估模式
model.eval()

# 定义一个函数来计算句子的BERT嵌入
def get_sentence_embedding(sentence):
    # 对句子进行编码
    encoded_input = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
    # 获取BERT的输出
    with torch.no_grad():
        output = model(**encoded_input)
    # 提取[CLS]标记的嵌入作为句子的嵌入
    sentence_embedding = output.last_hidden_state[:, 0, :]
    return sentence_embedding

# 定义两个句子
sentence_A = "今天的天气怎么样？"
sentence_B = "天气如何？"

# 计算两个句子的嵌入
embedding_A = get_sentence_embedding(sentence_A)
embedding_B = get_sentence_embedding(sentence_B)

# 将嵌入转换为numpy数组
embedding_A = embedding_A.numpy().flatten()
embedding_B = embedding_B.numpy().flatten()

# 计算余弦相似度
similarity = cosine_similarity([embedding_A], [embedding_B])[0][0]

print(f"Sentence A: {sentence_A}")
print(f"Sentence B: {sentence_B}")
print(f"Similarity: {similarity}")
