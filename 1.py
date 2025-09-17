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

genai.configure(api_key='AIzaSyBHccSmWckEEYteXKXBR4l9Y-TJJMnecn4',transport='rest')
model = genai.GenerativeModel('gemini-1.5-flash')

print('start')
response = model.generate_content(
'现在你是一个法律罪名判别器。在中国法律体系下，该案例犯了什么罪？用["罪1","罪2","罪3"……]的形式写出来，可以只有一个。/n案例：' + '成都市双流区人民检察院指控：2019年6月初，被告人胡江虚构可以办理成都信息工程学院学士学位证书的事实，与无法正常取得该校学士学位的学生被害人律九州商议好，以人民币16000元的价格帮助其办理学位证书，被告人胡江先行收取了人民币8000元。后被害人律九州表示“当兵，不需要再办理证书”并要求被告人胡江退钱，被告人胡江继续欺骗被害人律九州可以办证，并制作了虚假的印有律九州名字的成都信息工程学院学士学位证书拍照发给被害人律九州，称证书已办理好并要求其支付余下的人民币8000元，后被害人律九州发现被告人胡江发给其的学士学位照片有异并到公安机关咨询，民警遂在被告人胡江与被害人律九州约定见面的地点将其挡获。公诉机关认为，被告人胡江的行为构成诈骗罪，鉴于其无犯罪前科、自愿认罪认罚、退赔并取得谅解，建议判处有期徒刑六个月，缓刑一年，并处罚金人民币二千元。被告人胡江对指控事实、罪名及量刑建议均无异议，表示认罪认罚且签字具结，在开庭审理过程中亦无异议。其辩护人对指控事实、罪名及量刑建议均无异议'
    ,safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    }
    )
output = response.text
print(output)
    # print(call_with_stream(fact))