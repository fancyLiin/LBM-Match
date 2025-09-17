# -- coding:utf-8 --

from textrank4zh import TextRank4Keyword,TextRank4Sentence
import json
import os
def summary(fact):
    text = fact
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, lower=True, window=2)
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    filecontent = ''
    items = tr4s.get_key_sentences(num=5)
    sorted_items = sorted(items, key=lambda x: x['index'])
    for item in sorted_items:
        filecontent = filecontent + item.sentence + "。"
    filecontent = filecontent.replace('\n', '')
    return filecontent

items = []
# temp = 0
# # 假设你有一个包含多个JSON对象的字符串
# file_list = os.listdir(r'E:\Case_match\LeCaRDv2-main\candidate')
# for f1 in file_list:
#     condidateid = f1[:-5]
#     with open('datasets_p\\newcondidate\\'+ condidateid +'.json', 'w', encoding='utf-8') as file:
#         with open('LeCaRDv2-main\candidate\\' + condidateid + '.json', 'r', encoding='utf-8') as file:
#             context = json.load(file)
#             condi = {}
#             condi['cid'] = context['pid']
#             condi['reason'] = context['reason']
#             condi['article'] = context['article']
#             condi['charge'] = context['charge']
#             try:
#                 if len(context['reason']) > 1000:
#                     condi['reason'] = summary(context['reason'])
#                 # 将处理后的对象添加到列表中
#             except json.JSONDecodeError as e:
#                 print(f"错误：{e}")
#                 continue
#             with open('datasets_p\\newcondidate\\'+str(condi['cid'])+'.json', 'w', encoding='utf-8') as file:
#                 json.dump(condi, file, ensure_ascii=False)
#     temp = temp+1
#     if temp%100 == 0 :
#         output = f"{temp * 100:.2f}%"
#         print("已完成:"+output)
#
# print("处理完成")

with open(r'E:\Case_match\LeCaRDv2-main\query\query.json', 'r', encoding='utf-8') as file:
    for line in file:
        # 尝试解析每一行为JSON对象
        try:
            item = json.loads(line)
            # 检查fact的长度，并删除query键
            if len(item['fact']) > 1000:
                item['fact'] = summary(item['fact'])
            del item['query']
            # 将处理后的对象添加到列表中
            items.append(item)
        except json.JSONDecodeError as e:
            print(f"错误：{e}")
            continue

with open('datasets_p\dataquery.json', 'w', encoding='utf-8') as file:
    for item in items:
        # 将每个JSON对象转换为字符串并写入文件
        json_str = json.dumps(item, ensure_ascii=False)
        file.write(json_str + '\n')
print("处理完成，已修改data.json文件，并删除了query键。")


# items = []
#
# # 假设你有一个包含多个JSON对象的字符串
# # file_list = os.listdir(r'E:\Conversational_case_search\datasets\corpus')
# with open(r'E:\Case_match\LeCaRDv2-main\query\train_query.json', 'r', encoding='utf-8') as file:
#     for line in file:
#         # 尝试解析每一行为JSON对象
#         try:
#             item = json.loads(line)
#             # 检查fact的长度，并删除query键
#             if len(item['fact']) > 1000:
#                 item['fact'] = summary(item['fact'])
#             del item['query']
#             # 将处理后的对象添加到列表中
#             items.append(item)
#         except json.JSONDecodeError as e:
#             print(f"错误：{e}")
#             continue
#
# with open('datasets_p\data.json', 'w', encoding='utf-8') as file:
#     for item in items:
#         # 将每个JSON对象转换为字符串并写入文件
#         json_str = json.dumps(item, ensure_ascii=False)
#         file.write(json_str + '\n')
# print("处理完成，已修改data.json文件，并删除了query键。")

