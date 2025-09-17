import os
import json
import re

# 读取案例A的JSON数据集
def read_json_data_a(file_path):
    data_a = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_a.append(json.loads(line))
    return data_a

# 读取案例B的JSON数据集
def read_json_data_b(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# cid找对应         
def read_json_by_cid(cid, json_dir):
    json_files = [f for f in os.listdir(json_dir) if f.startswith(cid) and f.endswith('.json')]
    if not json_files:
        return []
    # 假设每个JSON文件只包含一个案例B的数据
    with open(os.path.join(json_dir, json_files[0]), 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# 主函数
def main():
    temp = 0
    data_a_path = 'datasets_p/dataquery.json'  # 案例A的JSON文件路径
    data_b_path = 'LeCaRDv2-main/candidate'  # 案例B的JSON文件路径
    labels_path = 'LeCaRDv2-main/label/test_relevence.trec'   # 标签文件的路径
    data_a = read_json_data_a(data_a_path)
    with open('GPT_data\\test.json', 'w', encoding='utf-8') as file: #可更改
        with open(labels_path, 'r', encoding='utf-8') as f:
            dict_array = []
            for line in f:
                temp = temp + 1
                context = {}
                parts = line.strip().split('\t')
                # if len(parts) == 4:
                #     id_, _, cid_, label = parts
                #     labels.append((id_, cid_, label))
                id1 = parts[0]
                cid1 = parts[2]
                label1 = parts[3]
                fact= ''
                for item in data_a:
                    if item['id'] == int(id1):
                        fact = item['fact']
                data_b = {}
                data_b = read_json_by_cid(str(cid1),data_b_path)
                labeltext = ""
                if label1 == str(0):
                    labeltext = "对于案例A和案例B，根据其与候选判决的匹配程度，我给出的评分是0分。该案例与候选判决没有任何相似之处，案件事实与判决结果完全不符，无法供该案例的当事人参考。案例文本与候选判决结果之间案情差异极大。"
                elif label1 == str(1):
                    labeltext = "对于案例A和案例B，根据其与候选判决的匹配程度，我给出的评分是1分。该案例与候选判决匹配程度一般,案件事实与判决结果存在一定出入,仅可供该案例的当事人参考。案例文本与候选判决结果之间案情相似之处偏少。"
                elif label1 == str(2):
                    labeltext = "对于案例A和案例B，根据其与候选判决的匹配程度，我给出的评分是2分。该案例与候选判决较为吻合,案件事实与判决结果基本符合引用规范,可供该案例的当事人进行匹配判断。案例文本与候选判决结果之间案情有一定的相似程度。"
                else:
                    labeltext = "对于案例A和案例B，根据其与候选判决的匹配程度，我给出的评分是3分。该案例与候选判决完全吻合,案件事实与判决结果均符合引用规范,可提供给该案例的当事人进行精准匹配判断。案例文本与候选判决结果之间案情最为相似。"
                items_with_numbers = [f"第{item}条" for i, item in enumerate(data_b['article'], start=1)]
                if type(data_b['charge']) == list:
                    items_with_numbers1 = [f"{item}" for i, item in enumerate(data_b['charge'], start=1)]
                else:
                    items_with_numbers1 =data_b['charge']
                result1 = " 、".join(items_with_numbers)
                result = " 、".join(items_with_numbers)
                context = {'instruction':"你是一个法律匹配模型，请为以下法律案例A，候选案例B，根据其与候选判决的匹配程度给出0到3的评分。",
                           'input':"案例A的详细情况是"+fact+"\n案例B违反了刑法"+result+",构成"+result1+"。详细情况是"+data_b['fact'],
                           'output':labeltext
                           }
                dict_array.extend([context])
                if temp % 500 == 0:
                    output = f"{temp*100 / 5000:.2f}%"
                    print("已完成:" + output)
                json_str = json.dumps(context, ensure_ascii=False)
                file.write(json_str + '\n')
if __name__ == '__main__':
    main()

# with open('datasets_p/data.json', 'r') as f:
#     for line in f:
#     # 尝试解析每一行为JSON对象
#             try:
#                 item = json.loads(line)
#                 items.append(item)
#             except json.JSONDecodeError as e:
#                 print(f"错误：{e}")
#                 continue
#
#     with open('datasets_p\data.json', 'w', encoding='utf-8') as file:
#         for item in items:
#             # 将每个JSON对象转换为字符串并写入文件
#             json_str = json.dumps(item, ensure_ascii=False)
#             file.write(json_str + '\n')
#     print("处理完成，已修改data.json文件，并删除了query键。")
#     case_data = json.load(f)
#
#     fact_description = case_b_data['案件事实描述']
#
#     # 提取违反的刑法条目
#     offense_articles = case_b_data['违反的刑法条目']
#
#     # 提取所犯的罪名
#     charge = case_b_data['所犯的罪名']
