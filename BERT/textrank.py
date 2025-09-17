#-*- encoding:utf-8 -*-
from __future__ import print_function
import json
from textrank4zh import TextRank4Sentence
tr4s = TextRank4Sentence()
tr4s1 = TextRank4Sentence()
temp = 0
with open('TR_test.json', 'w', encoding='utf-8') as f:
    with open('../QCtest.json', 'r', encoding='utf-8') as f1:
        for line in f1:
            temp =temp +1
            textcp = {}
            data = json.loads(line)
            text = data["qfact"]
            tr4s.analyze(text=text, lower=True, source = 'all_filters')
            qtext = ""
            for item in tr4s.get_key_sentences(num=3):
                qtext = qtext + item.sentence +"。" # index是语句在文本中位置，weight是权重
            text1 = data["cfact"]
            tr4s1.analyze(text=text1, lower=True, source = 'all_filters')
            ctext = ""
            for item in tr4s1.get_key_sentences(num=3):
                ctext = ctext + item.sentence +"。" # index是语句在文本中位置，weight是权重
            label = int(data["label"])
            textcp["q"] =qtext
            textcp["c"] = ctext
            textcp["label"] = label
            f.write(json.dumps(textcp,ensure_ascii=False) + '\n')
            if temp % 500 == 0:
                output = f"{temp * 100 / 5000:.2f}%"
                print("已完成:" + output)
