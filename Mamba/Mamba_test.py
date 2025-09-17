import json
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, MambaForCausalLM, MambaConfig

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

# 确保模型在评估模式下运行
model.eval()

# 读取JSON数据集
dataset = [json.loads(line)
        for line in open('test.json', 'r', encoding='utf-8')]

# 定义预测函数
def predict(A_case, B_candidate_text, B_charge, B_law_clause):
    # 构建输入文本
    input_text = f"A案例: {A_case} B候选文本: {B_candidate_text} B罪名: {B_charge} B涉及的法条: {B_law_clause}"

    # 对输入文本进行编码
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # 使用模型进行推理
    with torch.no_grad():  # 确保不会计算梯度
        outputs = model(inputs)

    # 获取输出日志概率的对数
    log_probs = outputs.logits

    # 我们需要定义一个方法来将模型输出转换为0-3的整数匹配程度
    # 这里的转换方法需要根据模型的输出和具体任务来定义
    # 以下是一个示例方法，可能需要根据实际情况进行调整
    match_score = torch.argmax(log_probs, dim=-1).item()

    # 将匹配程度映射到0-3的整数
    match_level = match_score % 4  # 假设模型输出有4个类别，这里简化处理

    return match_level

with open('legal.json', 'r', encoding='utf-8') as file:
    json_content = json.load(file)
# 进行推理并收集预测结果
true_labels = []
predicted_labels = []
for data in dataset:
    temp = 0
    A_case = data['query']
    B_candidate_text = data['candidate']
    B_charge = str(data['article'])
    B_law_clause = str([json_content[id]['context'] for id in data['charge'] if id in json_content])
    true_label = data['label']
    predicted_label = predict(A_case, B_candidate_text, B_charge, B_law_clause)
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)
    temp = temp + 1
    if temp % 500 == 0:
        output = f"{temp * 100 / 5000:.2f}%"
        print("已完成:" + output)
# 计算准确率、召回率和F1分数
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
print(f"准确率: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1分数: {f1:.2f}")
#130M准确率: 0.21 召回率: 0.25 F1分数: 0.11
#2B准确率: 0.24召回率: 0.22F1分数: 0.16
#Accuracy: 0.6679874869655892
# Mean AUC-ROC: 0.6841999148004835
# Mean PR-AUC: 0.364041283798063