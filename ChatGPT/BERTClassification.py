# -- coding:utf-8 --
import json
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def read_json_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def calculate_accuracy(predicted, actual):
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    return correct / len(actual)

def similarity_to_score(similarity):
    # 简单规则：可以按需微调
    if similarity > 0.85:
        return 3
    elif similarity > 0.7:
        return 2
    elif similarity > 0.5:
        return 1
    else:
        return 0

if __name__ == '__main__':
    data_path = '../GPT_data/test.json'  # 路径视项目而定
    data = read_json_data(data_path)

    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')  # 可换更强的模型

    truelabels = []
    predictedlabels = []

    for item in tqdm(data[:100]):  # 调整数量以节省时间
        input_text = item['input']
        output_text = item['output']

        # 从output中提取真实标签
        match = re.search(r'[0123]', output_text)
        truelabel = int(match.group()) if match else 0
        truelabels.append(truelabel)

        # 分离A和B（假设有“案例A的详细情况是”和“案例B”）
        try:
            a_text = re.search(r'案例A的详细情况是(.*?)案例B', input_text, re.S).group(1).strip()
            b_text = re.search(r'案例B(.*)', input_text, re.S).group(1).strip()
        except:
            a_text, b_text = input_text[:200], input_text[200:]  # 兜底处理

        # 计算余弦相似度
        emb = model.encode([a_text, b_text])
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]

        predlabel = similarity_to_score(sim)
        predictedlabels.append(predlabel)

    with open('truelabel_bert.json', 'w', encoding='utf-8') as f:
        json.dump(truelabels, f, ensure_ascii=False)
    with open('outputlabel_bert.json', 'w', encoding='utf-8') as f:
        json.dump(predictedlabels, f, ensure_ascii=False)

    acc = calculate_accuracy(predictedlabels, truelabels)
    print(f'Accuracy: {acc:.4f}')