#-*-coding: UTF-8 -*-
from http import HTTPStatus
import dashscope
from dashscope import Generation
import json
import ast
import pathlib
import textwrap

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from IPython.display import display
from IPython.display import Markdown

genai.configure(api_key='AIzaSyBHccSmWckEEYteXKXBR4l9Y-TJJMnecn4',transport="rest")
model = genai.GenerativeModel('gemini-1.5-flash')
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

dashscope.api_key = 'sk-93b641a143d94b21bb002180372c8f92'

def read_json_data_a(file_path):
    data_a = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_a.append(json.loads(line))
    return data_a

def call_with_stream(fact):

    output= ''
    messages = [
        {'role': 'user', 'content': '现在你是一个法律罪名判别器。在中国法律体系下，该案例犯了什么罪？用["罪1","罪2","罪3"……]的形式写出来，可以只有一个。/n案例：'+fact}]
    responses = Generation.call("qwen-max",
                                messages=messages,
                                result_format='message',  # 设置输出为'message'格式
                                stream=True, # 设置输出方式为流式输出
                                incremental_output=True  # 增量式流式输出
                                )
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            output = output + response.output.choices[0]['message']['content']

        else:
            output=''
    return output

    # for response in responses:
    #     if response.status_code == HTTPStatus.OK:
    #         result = response.output.choices[0]['message']['content']
    #     else:
    #         result = response.code + response.message
    #     return result


def append_to_json_file(file_path, data):
    """将数据追加到指定的JSON文件中"""
    with open('test.json', 'a', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')  # 在JSON对象之间添加换行符以便于阅读

def process_output(output):
    """处理模型输出，确保其符合预期格式"""
    if output.startswith('[') and output.endswith(']'):
        return ast.literal_eval(output)
    elif '[' in output and ']' in output:
        start_index = output.find('[')
        end_index = output.find(']')
        if start_index != -1 and end_index != -1 and end_index > start_index + 1:
            return ast.literal_eval(output[start_index:end_index + 1])
    else:
        return []

if __name__ == '__main__':
    data_a_path = 'datasets_p/dataquery.json'
    query = read_json_data_a(data_a_path)

    output_file_path = 'querycharge.json'

    # 初始化或清空输出文件
    open(output_file_path, 'w', encoding='utf-8').close()
    temp = 0
    x = 149
    for i in query[(x-1):x]:
        temp = temp+1# 假设您想打印的是id而不是累加的计数
        fact = i['fact']
        output = call_with_stream(fact)
        if output == '':
            output = model.generate_content(
                '现在你是一个法律罪名判别器。在中国法律体系下，该案例犯了什么罪？用["罪1","罪2","罪3"……]的形式写出来，可以只有一个,也可以是多个。/n案例：' + fact,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                }
            ).text

        # 处理并验证输出
        charges = process_output(output)
        print(str(temp)+'/n'+output)
        # 直接将处理结果写入文件
        # charge_entry = {"id": i['id'], "charge": charges}
        # append_to_json_file(output_file_path, charge_entry)



# if __name__ == '__main__':
#     data_a_path = 'datasets_p/dataquery.json'  # 案例A的JSON文件路径
#     query = read_json_data_a(data_a_path)
#     temp =  0
#     for i in query:
#         temp=temp+1
#         print(temp)
#         fact = i['fact']
#         id = int(i['id'])
#         result = {}
#         output = call_with_stream(fact)
#         if output=='':
#             output = model.generate_content('现在你是一个法律罪名判别器。在中国法律体系下，该案例犯了什么罪？用["罪1","罪2","罪3"……]的形式写出来，可以只有一个,也可以是多个。/n案例：'+fact,
#                                               safety_settings={
#                                                   HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#                                                   HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#                                                   HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#                                                   HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
#                                               }
#                                               ).text
#         print(output)
#
#             # 检查输出是否完全匹配[xxxx]的格式
#         if output.startswith('[') and output.endswith(']'):
#             output = ast.literal_eval(output)
#             # 检查输出中是否包含[xxx]的部分
#         elif '[' in output and ']' in output:
#             # 提取第一个符合条件的[xxx]部分，假设格式正确，只提取第一个匹配项
#             start_index = output.find('[')
#             end_index = output.find(']')
#             if start_index != -1 and end_index != -1 and end_index > start_index + 1:
#                 output = ast.literal_eval(output[start_index:end_index + 1])
#         else:
#             # 不满足上述条件，返回空列表
#             output =  []
#         result[id] = output
#
#     json_array = [
#         {"id": id, "charge": output}
#         for id, output in result.items()
#     ]
#
#     # 将JSON数组保存到文件
#     with open('querycharge.json', 'w', encoding='utf-8') as file:
#         json.dump(json_array, file, ensure_ascii=False)
#     # print(call_with_stream(fact))