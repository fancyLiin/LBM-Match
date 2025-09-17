import argparse
import torch
import re
import json
from sklearn.metrics import precision_recall_fscore_support
from model import MODE


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser()
    # 模型配置参数
    parser.add_argument("--device", type=str, default="0", help="")
    parser.add_argument("--mode", type=str, default="glm3", help="")
    parser.add_argument("--model_path", type=str, default="output_dir/", help="")
    parser.add_argument("--max_length", type=int, default=2048, help="")
    parser.add_argument("--do_sample", type=bool, default=True, help="")
    parser.add_argument("--top_p", type=float, default=0.8, help="")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    return parser.parse_args()


def predict_one_sample(instruction, input, model, tokenizer, args):
    """
    使用模型对单个样本进行预测。
    """
    result, _ = model.chat(tokenizer, instruction + input, max_length=args.max_length, do_sample=args.do_sample,
                           top_p=args.top_p, temperature=args.temperature)
    return result


def test_model(test_data_path, model, tokenizer, args):
    """
    使用模型对测试集进行预测，并计算精确值、召回值和F1分数。
    """
    test_data = []
    test_labels = []

    # 读取JSON格式的数据集文件
    with open(test_data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            score = re.search(r'\d+', data['output'])
            r = int(score.group(1))
            test_data.append((data['instruction']+data['input']))  # 假设"某某内容"是你的指令
            test_labels.append(r)


    predictions = []
    for instruction, input in test_data:
        text = predict_one_sample(instruction, input, model, tokenizer, args)
        score = re.search(r'\d+', text)
        r=int(score.group(1))
        if isinstance(r, int) and 0 <= r <= 3:
            r = r
        else:
            r = 0
        # 将模型输出转换为0-3的整数
        predictions.append(r)

    # 计算精确值、召回值和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='macro',
                                                               zero_division=1)
    return precision, recall, f1


if __name__ == '__main__':
    args = parse_args()
    # 加载模型和分词器
    model = MODE[args.mode]["model"].from_pretrained(args.model_path, device_map="cuda:{}".format(args.device),
                                                     torch_dtype=torch.float16)
    tokenizer = MODE[args.mode]["tokenizer"].from_pretrained(args.model_path)

    test_data_path = 'path_to_your_test_dataset.json'

    # 运行测试函数并打印结果
    precision, recall, f1 = test_model(test_data_path, model, tokenizer, args)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")