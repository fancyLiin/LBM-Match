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
    data_a_path = '../datasets_p/dataquery.json'  # 案例A的JSON文件路径
    data_b_path = '../datasets_p/newcondidate'  # 案例B的JSON文件路径
    labels_path = '../LeCaRDv2-main/label/relevence.trec'   # 标签文件的路径
    data_a = read_json_data_a(data_a_path)
    with open('datasets\\train.json', 'w', encoding='utf-8') as file: #可更改
        with open(labels_path, 'r', encoding='utf-8') as f:
            dict_array = []
            for line in f:
                temp = temp + 1
                context = {}
                parts = line.strip().split('\t')
                id1 = parts[0]
                cid1 = parts[2]
                label1 = parts[3]
                fact= ''
                for item in data_a:
                    if item['id'] == int(id1):
                        fact = item['fact']
                data_b = {}
                data_b = read_json_by_cid(str(cid1),data_b_path)
                context = {'query': fact,
                           'candidate': data_b['reason'],
                           'article': data_b['article'],
                           'charge': data_b['charge'],
                           'label': label1
                           }
                dict_array.extend([context])
                if temp % 500 == 0:
                    output = f"{temp*100 / 25000:.2f}%"
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
