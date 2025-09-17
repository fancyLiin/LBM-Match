from sklearn.metrics import accuracy_score
import json
# 假设的真值和预测列表
y_true = []
y_pred = []
with open('truelabel.json', 'r', encoding='utf-8') as file:
    # 读取整个文件的内容
    json_data = file.read()
    y_true = json.loads(json_data)

with open('outputlabel.json', 'r', encoding='utf-8') as file:
    # 读取整个文件的内容
    json_data = file.read()
    y_pred = json.loads(json_data)
# 计算准确率
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)