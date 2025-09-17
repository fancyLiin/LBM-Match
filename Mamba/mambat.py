import json
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, MambaForCausalLM, MambaConfig

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

# 确保模型在评估模式下运行
model.eval()
input = tokenizer.encode('案例A的详细情况是瑞安市人民检察院指控，2011年4月份开始，被告人周小金伙同赵某、张某、朱某乙（均已判）等人在瑞安市汀田街道凤岙山的老人活动中心、半山腰坟地、山脚下等地开设赌场，召集他人以扑克牌九的形式进行赌博，按庄家赢钱额的5%抽取头薪，同时雇佣他人接送赌客、望风、打皮等，其中被告人车小云负责望风。该赌场开设20余日，共抽取头薪30余万元。被告人周小金辩称，与赵某等人开设赌场属实，但赌场开设时间只有五、六天，抽取头薪没有30万元；被告人车小云辩称，没有为赌场望风，也没有去过赌场，没有从赌场中分到钱。辩护人戴胜平的辩护意见是，1、起诉书认定被告人周小金开设赌场的时间及抽取头薪金额证据不足，证人证言有相互矛盾的地方，头薪金额应该没有30万元。2、被告人周小金前罪也是开设赌场，属于同一种罪，并罚时要尽量从轻。3、被告人周小金的认罪态度比较好，供述是稳定的。综上，建议对其从轻处罚。辩护人周燕青的辩护意见是，起诉书指控被告人车小云犯开设赌场罪证据不足，证人证言不能形成证据链条，证实被告人在赌场望风的股东只有赵某，周小金没有证明，被告人车小云是被谁雇佣不明确。\n案例B违反了刑法第303条 、第277条 、第25条 、第26条 、第27条 、第69条 、第65条 、第67条 、第68条 、第72条 、第64条,构成第303条 、第277条 、第25条 、第26条 、第27条 、第69条 、第65条 、第67条 、第68条 、第72条 、第64条。详细情况是本院认为，被告人洪良、欧阳进仕、戴方中、林绍祥、吴全洁、蔡士丰、黄贤朋、朱世楷以营利为目的，结伙开设赌场，其中被告人洪良、欧阳进仕属情节严重；被告人洪良又以暴力、威胁方法阻碍国家机关工作人员依法执行职务，其行为均已触犯刑律，被告人洪良分别构成开设赌场罪、妨害公务罪；被告人欧阳进仕、戴方中、林绍祥、吴全洁、蔡士丰、黄贤朋、朱世楷构成开设赌场罪。公诉机关指控的罪名均成立。', return_tensors="pt")
print(input.size())
print(input.shape)

# Epoch 1/10, Loss: 1.1635537350448333
# Epoch 2/10, Loss: 0.7584647948894705
# Epoch 3/10, Loss: 0.6675195334670699
# Epoch 4/10, Loss: 0.6068924902596365
# Epoch 5/10, Loss: 0.5594560312970139
# Epoch 6/10, Loss: 0.5096893363267939
# Epoch 7/10, Loss: 0.4720047614046028
# Epoch 8/10, Loss: 0.42834652753395136
# Epoch 9/10, Loss: 0.39522163905153607
# Epoch 10/10, Loss: 0.3546664775890088
# Training completed.
# Model saved.
# Traceback (most recent call last):
#   File "/root/Mamba2.py", line 106, in <module>
#     outputs = model(batch_embeddings).squeeze(1)
# NameError: name 'test_loader' is not defined. Did you mean: 'dataloader'?
# root@autodl-container-817c4686dc-a6326ce1:~# python Test.py
# Traceback (most recent call last):
#   File "/root/Test.py", line 30, in <module>
#     Temblist1 = Temblist1 + [qvec]
# NameError: name 'qvec' is not defined. Did you mean: 'Tqvec'?
# root@autodl-container-817c4686dc-a6326ce1:~# python Test.py
# 读取完成
# Traceback (most recent call last):
#   File "/root/Test.py", line 44, in <module>
#     dataset = TensorDataset(combined_embeddings, labels)
# NameError: name 'labels' is not defined. Did you mean: 'Tlabels'?
# root@autodl-container-817c4686dc-a6326ce1:~# python Test.py
# 读取完成
# Model loaded.
# Accuracy: 0.8859228362877998
# Traceback (most recent call last):
#   File "/root/Test.py", line 80, in <module>
#     all_preds_prob = np.array(all_preds_prob)
# NameError: name 'np' is not defined. Did you mean: 'nn'?
# root@autodl-container-817c4686dc-a6326ce1:~# python Test.py
# 读取完成
# Model loaded.
# Accuracy: 0.8859228362877998
# Mean AUC-ROC: 0.9772555274161279
# Mean PR-AUC: 0.9332949811901103