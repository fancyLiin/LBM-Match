# -- coding:utf-8 --
import numpy as np
import os
from tqdm import tqdm
import json
from openai import OpenAI
def main():
    temp = 0
    datatest = 'QCtest.json'
    datatrain = 'QCresult.json'

client = OpenAI(api_key = "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")
def ChatGPTembedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding
def append_to_file(file_path, data):
    """追加数据到指定文件"""
    with open(file_path, 'a', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')  # 在每个JSON对象之后添加换行符
datatest ='../QCresult.json'
encoded_file_path = 'Trian_ChatGPT_datavec.json'
# 清空或创建文件，避免在追加模式下文件已有内容
open(encoded_file_path, 'w', encoding='utf-8').close()
# open(cossim_file_path, 'w', encoding='utf-8').close()
with open(datatest, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        data_dict = json.loads(line)
        qfact = data_dict["qfact_x"]
        cfact = data_dict["cfact_x"]
        qcharge = data_dict["qcharge_x"]
        ccharge = data_dict["ccharge_x"]
        vec1 = ChatGPTembedding(qfact)
        vec2 = ChatGPTembedding(cfact)
        vec3 = ChatGPTembedding(qcharge)
        vec4 = ChatGPTembedding(ccharge)
        # vec2 = model.encode(cfacts)
        # vec3 = model.encode(qcharge)
        # vec4 = model.encode(ccharge)
        # vec5 = model.encode(qresult)
        # vec6 = model.encode(cresult)
        #             fact_cos_sim = vec1 @ vec2.T
        # charge_cos_sim = vec5 @ vec6.T
        # if

        # 准备数据字典
        encoded_data = {
            # 'qid': data_dict['qid'],
            # 'cid': data_dict['cid'],
            # 'qfact_vec': vec1.tolist(),
            # 'cfact_vec': vec2.tolist(),
            # 'qcharge_vec': vec3.tolist(),
            # 'ccharge_vec': vec4.tolist(),
            # 'label': data_dict['label']
            'qid': data_dict['qid'],
            'cid': data_dict['cid'],
            'qfact_vec': vec1,
            'cfact_vec': vec2,
            'qcharge_vec': vec3,
            'ccharge_vec': vec4,
            'label': data_dict['label_x']
        }
        #             cossim_data = {
        #             #                 'qid': data_dict['qid'],
        #             #                 'cid': data_dict['cid'],
        #             #                 'fact_cos_sim': fact_cos_sim.tolist(),
        #             #                 'cfact_cos_sim': ctoq_cos_sim.tolist(),
        #             #                 'qfact_cos_sim': qtoc_cos_sim.tolist(),
        #             #                 'charge_cos_sim': charge_cos_sim.tolist(),
        #             #                 'label': data_dict['label']
        #                             'qid': data_dict['qid'],
        #                             'cid': data_dict['cid'],
        #                             'charge_cos_sim': charge_cos_sim.tolist(),
        #                             'label': data_dict['label']
        #                             }

        # 追加数据到文件
        append_to_file(encoded_file_path, encoded_data)
        # append_to_file(cossim_file_path, cossim_data)
# def main():
#     temp = 0
#     datatest = 'QCtest.json'
#     datatrain = 'QCresult.json'
#     model = FlagModel('BAAI/bge-large-zh-v1.5',
#                       query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
#                       use_fp16=True)
#
#     encoded_file_path = 'Train_all_datavec.json'
#     # cossim_file_path = 'Test_c_datacossim.json'
#
#     # 清空或创建文件，避免在追加模式下文件已有内容
#     open(encoded_file_path, 'w', encoding='utf-8').close()
#     # open(cossim_file_path, 'w', encoding='utf-8').close()
#
#     with open(datatrain, 'r', encoding='utf-8') as f:
#         for line in f:
#             temp += 1
#             data_dict = json.loads(line)
#             qfact1 = data_dict["qfact_x"]
#             cfact1 = data_dict["cfact_x"]
#             qfacts, cfacts = [], []
#             qfact = qfact1.split("。")
#             # 由于split()会删除分隔符，如果句号是你句子的一部分，你可能需要重新添加它们
#             qfacts = [qfact + "。" for qfact in qfacts if qfact]
#
#             cfact = cfact1.split("。")
#             cfacts = [cfact + "。" for cfact in cfacts if cfact]
#             qcharge = data_dict["qcharge_x"]
#             ccharge = data_dict["ccharge_x"]
#
#             qresult = ','.join(qcharge)
#             cresult = ','.join(ccharge)
#             vec1, vec2 = [], []
#             for qfact in qfacts:
#                 vectemp = model.encode(qfact).tolist()
#                 vec1 = vec1 + [vectemp]
#             for cfact in cfacts:
#                 vectemp = model.encode(cfact).tolist()
#                 vec2 = vec2 + [vectemp]
#             # vec1 = model.encode(qfacts)
#             # vec2 = model.encode(cfacts)
#             # vec3 = model.encode(qcharge)
#             # vec4 = model.encode(ccharge)
#             vec5 = model.encode(qresult)
#             vec6 = model.encode(cresult)
#             #             fact_cos_sim = vec1 @ vec2.T
#             # charge_cos_sim = vec5 @ vec6.T
#             # if
#
#             # 准备数据字典
#             encoded_data = {
#                 # 'qid': data_dict['qid'],
#                 # 'cid': data_dict['cid'],
#                 # 'qfact_vec': vec1.tolist(),
#                 # 'cfact_vec': vec2.tolist(),
#                 # 'qcharge_vec': vec3.tolist(),
#                 # 'ccharge_vec': vec4.tolist(),
#                 # 'label': data_dict['label']
#                 'qid': data_dict['qid'],
#                 'cid': data_dict['cid'],
#                 'qfact_vec': vec1,
#                 'cfact_vec': vec2,
#                 'qcharge_vec': vec5.tolist(),
#                 'ccharge_vec': vec6.tolist(),
#                 'label': data_dict['label_x']
#             }
#             #             cossim_data = {
#             #             #                 'qid': data_dict['qid'],
#             #             #                 'cid': data_dict['cid'],
#             #             #                 'fact_cos_sim': fact_cos_sim.tolist(),
#             #             #                 'cfact_cos_sim': ctoq_cos_sim.tolist(),
#             #             #                 'qfact_cos_sim': qtoc_cos_sim.tolist(),
#             #             #                 'charge_cos_sim': charge_cos_sim.tolist(),
#             #             #                 'label': data_dict['label']
#             #                             'qid': data_dict['qid'],
#             #                             'cid': data_dict['cid'],
#             #                             'charge_cos_sim': charge_cos_sim.tolist(),
#             #                             'label': data_dict['label']
#             #                             }
#
#             # 追加数据到文件
#             append_to_file(encoded_file_path, encoded_data)
#             # append_to_file(cossim_file_path, cossim_data)
#
#             if temp % 400 == 0:
#                 output = f"{temp*100 / 1==20000:.2f}%"
#                 print("已完成:" + output)
#
#
# if __name__ == '__main__':
#     main()