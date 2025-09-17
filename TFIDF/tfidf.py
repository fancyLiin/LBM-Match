# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jiojio
import json


def calculate_similarity(text1, text2):
    # 将文本分割成单词
    def tokenize(text):
        return text.split()

    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer(tokenizer=tokenize)

    # 计算TF-IDF向量
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # 计算余弦相似度
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity

def remove_punc(string):
    biaodian = '''!()-[]{};:'"\,<>./?@#$%^&*_~！？。，、；：“”『』（）《》〈〉「」﹃﹄‘’“”〝〞〔〕〖〗〘〙〚〛～]'''
    for i in string:
        if i in biaodian:
            string = string.replace(i, "")
    return string


with open('../QCtest.json', 'r', encoding='utf-8') as f:
    list = []
    result1=[]
    result2=[]
    temp=0
    for line in f:
        temp = temp + 1
        if temp % 50 == 0:
            output = f"{temp * 100 / 5000:.2f}%"
            print("已完成:" + output)
        item = json.loads(line.strip())
        jiojio.init()
        text1 = jiojio.cut(item['qfact'])
        text2 = jiojio.cut(item['cfact'])
        result1 = [remove_punc(i) for i in text1]
        Fresult1 = [item for item in result1 if item]
        result2 = [remove_punc(i) for i in text2]
        Fresult2 = [item for item in result2 if item]
        separated_string1 = ' '.join(Fresult1)
        separated_string2 = ' '.join(Fresult2)
        # 计算相似度
        similarity = calculate_similarity(separated_string1, separated_string2)
        list = list+[similarity]
    json_string = json.dumps(list)
    with open('cos_tfidf_test.json', 'w') as file:
        json.dump(list, file)