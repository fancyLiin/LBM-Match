import re
import json

# 读取刑法TXT文件
with open('legal.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    # 初始化列表，用于存储拆分后的刑法条文
    sections = []

    # 遍历文件内容，拆分刑法条文
    current_id = None
    current_context = []
    truecontext= ''
    context=''
    n=0
    for index, line in enumerate(lines, start=1):
        line = line.lstrip()
        # 匹配条文编号，包括中文形式，并且前面有“第”字
        match1 = re.match(r'(第[零一二三四五六七八九十百千万]+章)', line)
        if match1:
            continue
        match = re.match(r'(第[零一二三四五六七八九十百千万]+条(?!之))', line)
        if match:
            n = n + 1
            current_id = match.group(1)
            if n == 1 :
                context = ''
                sections.append({
                    'id': n,
                    'context': ''.join(line)
                })
            if n != 1:
                truecontext = context
                context = line
                if n != 2:
                    sections.append({
                        'id': n-1,
                        'context': ''.join(truecontext)
                    })
                print(truecontext)
                print(n)
            # 清除当前内容列表
        else:
            context = context + line
            # 如果没有找到编号，则跳过
            continue
        # 当到达最后一条时，保存当前条文
sections.append({
                'id': 452,
                'context': ''.join('第四百五十二条　本法自1997年10月1日起施行。')
            })


# 将列表保存为JSON文件
with open('刑法条文.json', 'w', encoding='utf-8') as file:
    json.dump(sections, file, ensure_ascii=False, indent=4)
