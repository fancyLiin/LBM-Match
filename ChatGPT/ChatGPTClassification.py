# -- coding:utf-8 --
import numpy as np
import os
from tqdm import tqdm
import json
import re
from openai import OpenAI
def main():
    temp = 0
    datatest = 'QCtest.json'
    datatrain = 'QCresult.json'

# client = OpenAI(api_key = "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")
client = OpenAI(api_key = "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")


def append_to_file(file_path, data):
    """追加数据到指定文件"""
    with open(file_path, 'a', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')  # 在每个JSON对象之后添加换行符
datatest ='../QCtest.json'

# open(cossim_file_path, 'w', encoding='utf-8').close()
def read_json_data_a(file_path):
    data_a = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_a.append(json.loads(line))
    return data_a


def calculate_accuracy(predicted, actual):
    if len(predicted) != len(actual):
        raise ValueError("The two lists must have the same length.")

    correct = sum(1 for pred, act in zip(predicted, actual) if pred == act)

    return correct / len(predicted)
if __name__ == '__main__':
    data_a_path = '../GPT_data/test.json'
    query = read_json_data_a(data_a_path)
    # 初始化或清空输出文件
    temp = 0
    truelabellist=[]
    outputlabellist=[]
    for i in tqdm(query[:100]):
        temp = temp+1# 假设您想打印的是id而不是累加的计数
        input = i['input']
        pattern = r'[0123]'
        labeltrue = re.search(pattern, i['output'])
        if labeltrue:
            truelabel = int(labeltrue.group())
        else:
            truelabel = 0
        truelabellist = truelabellist +[truelabel]
        completion = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": "你是一个法律匹配模型."},
                {"role": "user", "content": "请为以下法律案例A，候选案例B，根据AB之间的匹配程度给出0到3的评分。（0分完全不吻合，1分只有一点，2分一般，3分较为吻合）"+input}
                    ]
            )
        output = completion.choices[0].message
        # 编写正则表达式，寻找0到3之间的整数
        # 使用search找到第一个匹配
        match = re.search(pattern, str(output))
        if match:
            label = int(match.group())
        else:
            label = 0
        outputlabellist = outputlabellist + [label]
        json_true = json.dumps(truelabellist)
        json_output = json.dumps(outputlabellist)
        with open('truelabel_4o.json', 'w') as f:
            json.dump(truelabellist, f)
        with open('outputlabel_4o.json', 'w') as f:
            json.dump(outputlabellist, f)
    print(calculate_accuracy(outputlabellist,truelabellist))