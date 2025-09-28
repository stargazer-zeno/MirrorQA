import jsonlines
import os
import json
INFERENCE_DIR = "xxx"
REF_DIR = "xxx"
DATA_PATH = "xxx"
ORDER_PATH = "xxx"
DST_PATH = "xxx"

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def to_jsonl(data, file_path):
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(data)

def is_answer(s):
    patterns1 = ['a', 'b', 'c', 'd']
    patterns2 = ['a.', 'b.', 'c.', 'd.']
    for pattern in patterns1:
        if s == pattern: return True
    if len(s) < 15:
        for pattern in patterns2:
            if s.startswith(pattern): return True
    return False

def match_pattern(s):
    option_patterns_4 = {'a': ['left front', 'left fore'], 'b': ['right front', 'right fore'], 'c': ['left rear', 'left hind'], 'd': ['right rear', 'right hind']}
    option_patterns_2 = {'a': ['left'], 'b': ['right']}
    for key in option_patterns_4:
        for op in option_patterns_4[key]:
            if op in s:
                return key
    for key in option_patterns_2:
        for op in option_patterns_2[key]:
            if op in s:
                return key

def parse_pred(pred):
    pred = pred.lower()
    pred = pred.split('\n')[-1].strip()
    answer_patterns = [
        "**answer**:",
        "**answer**",
        "*answer*:",
        "**answer:**",
        "answer is:",
        "answer is",
        "answer:",
    ]
    for answer_pattern in answer_patterns:
        if answer_pattern in pred:
            pred = pred.split(answer_pattern)[-1].strip()
    if is_answer(pred):
        return pred
    for t in range(20, 250, 10):
        if len(pred) > t:
            partial = pred[-t:]
        else:
            partial = pred
            match = match_pattern(partial)
            if match: return match
            break
        match = match_pattern(partial)
        if match: return match
    return ''

def judge(data, ref):
    for d in data:
        pred = parse_pred(d['pred'])
        gold = d['gold'].lower()
        if pred: extracted_ans = pred[0]
        else: extracted_ans = ''
        d['extracted_ans'] = extracted_ans
        d['correct'] = 1 if extracted_ans == gold else 0

    for d in ref:
        pred = parse_pred(d['pred'])
        gold = d['gold'].lower()
        if pred: extracted_ans = pred[0]
        else: extracted_ans = ''
        d['extracted_ans'] = extracted_ans
        d['correct'] = 1 if extracted_ans == gold else 0

def cal_acc(inf, ref):
    inf_idx = 0
    num_correct = 0
    for d in ref:
        circular_list = [d]
        while inf_idx < len(inf):
            if inf[inf_idx]['image'] == d['image']:
                circular_list.append(inf[inf_idx])
                inf_idx += 1
            else:
                break
        assert len(circular_list) == len(d['options'])
        if len([x for x in circular_list if x['correct']]) == len(circular_list):
            num_correct += 1
    return {'acc': num_correct / len(ref)}

def eval_one(model):
    files = sorted(os.listdir(INFERENCE_DIR))
    ref_files = sorted(os.listdir(REF_DIR))
    file = f"{model}.jsonl"
    assert file in files
    inf = read_jsonl(os.path.join(INFERENCE_DIR, file))
    try:
        ref_file = f"{model}_run0.jsonl"
        assert ref_file in ref_files
    except:
        ref_file = f"{model}.jsonl"
        assert ref_file in ref_files
    ref = read_jsonl(os.path.join(REF_DIR, ref_file))
    judge(inf, ref)
    metrics = cal_acc(inf, ref)
    return metrics

if __name__ == '__main__':
    results = {}
    with open(ORDER_PATH, 'r') as f:
        models = [item.strip() for item in f.readlines() if item]
    for model in models:
        results[model] = eval_one(model)
    with open(DST_PATH, 'w') as f:
        json.dump(results, f, indent=2)