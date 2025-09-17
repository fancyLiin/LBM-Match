#-*-coding: UTF-8 -*-
from http import HTTPStatus
import dashscope
from dashscope import Generation
import json
import ast
import pathlib
import textwrap
import re
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from IPython.display import display
from IPython.display import Markdown

genai.configure(api_key='AIzaSyBHccSmWckEEYteXKXBR4l9Y-TJJMnecn4',transport="rest")
model = genai.GenerativeModel('gemini-1.5-flash')

dashscope.api_key = 'sk-93b641a143d94b21bb002180372c8f92'

def read_json_data_a(file_path):
    data_a = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_a.append(json.loads(line))
    return data_a
def append_to_json_file(file_path, data):
    """将数据追加到指定的JSON文件中"""
    with open('test.json', 'a', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')  # 在JSON对象之间添加换行符以便于阅读

if __name__ == '__main__':
    data_a_path = '../GPT_data/test.json'
    query = read_json_data_a(data_a_path)
    output_file_path = 'querycharge.json'
    # 初始化或清空输出文件
    open(output_file_path, 'w', encoding='utf-8').close()
    temp = 0
    truelabellist=[]
    outputlabellist=[]
    for i in query[:100]:
        temp = temp+1# 假设您想打印的是id而不是累加的计数
        input = i['input']
        pattern = r'[0123]'
        labeltrue = re.search(pattern, i['output'])
        if labeltrue:
            truelabel = int(labeltrue.group())
        else:
            truelabel = 0
        truelabellist = truelabellist +[truelabel]

        output = model.generate_content(
            '你是一个法律匹配模型，请为以下法律案例A，候选案例B，根据AB之间的匹配程度给出0到3的评分。（0分完全不吻合，1分只有一点，2分一般，3分较为吻合）'+input,
            safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
            ).text
        # 编写正则表达式，寻找0到3之间的整数

        # 使用search找到第一个匹配
        match = re.search(pattern, output)
        print(temp)
        if match:
            label = int(match.group())
        else:
            label = 0
        outputlabellist = outputlabellist + [label]
        json_true = json.dumps(truelabellist)
        json_output = json.dumps(outputlabellist)
        with open('truelabel_1.json', 'w') as f:
            json.dump(truelabellist, f)
        with open('outputlabel_1.json', 'w') as f:
            json.dump(outputlabellist, f)
        print("已完成"+str(temp))
