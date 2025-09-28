import jsonlines
import os
import json
INFERENCE_DIR = "xxx"
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

def judge(data):
    for d in data:
        pred = parse_pred(d['pred'])
        gold = d['gold'].lower()
        if pred: extracted_ans = pred[0]
        else: extracted_ans = ''
        d['extracted_ans'] = extracted_ans
        d['correct'] = 1 if extracted_ans == gold else 0

def init_cnt(num_options):
    d = {}
    for i in range(num_options):
        d[chr(ord('a') + i)] = {'tp': 0, 'fp': 0, 'fn': 0}
    return d

def cal_prf(inf):
    counter = {'2': init_cnt(2), '4': init_cnt(4)}
    num_2, num_4 = 0, 0
    for d in inf:
        if len(d['options']) == 2:
            num_2 += 1
        elif len(d['options']) == 4:
            num_4 += 1
        if not d['extracted_ans']:
            counter[str(len(d['options']))][d['gold'].lower()]['fn'] += 1
        else:
            if d['correct']:
                counter[str(len(d['options']))][d['extracted_ans']]['tp'] += 1
            else:
                counter[str(len(d['options']))][d['extracted_ans']]['fp'] += 1
                counter[str(len(d['options']))][d['gold'].lower()]['fn'] += 1
    def compute_macro_prf(cnt, num_options):
        ps, rs, f1s = [], [], []
        for c in cnt.values():
            tp, fp, fn = c['tp'], c['fp'], c['fn']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            ps.append(p)
            rs.append(r)
            f1s.append(f1)
        return sum(ps)/num_options, sum(rs)/num_options, sum(f1s)/num_options

    p_2, r_2, f1_2 = compute_macro_prf(counter['2'], 2)
    p_4, r_4, f1_4 = compute_macro_prf(counter['4'], 4)

    total = num_2 + num_4
    pw = (p_2 * num_2 + p_4 * num_4) / total
    rw = (r_2 * num_2 + r_4 * num_4) / total
    fw = (f1_2 * num_2 + f1_4 * num_4) / total
    f_pr = (2 * pw * rw / (pw + rw)) if (pw + rw) > 0 else 0.0

    return {
        '2': {'p': p_2, 'r': r_2, 'f1': f1_2},
        '4': {'p': p_4, 'r': r_4, 'f1': f1_4},
        'weighted': {'p': pw, 'r': rw, 'f1': fw},
        'final_f1': f_pr
    }

def init_cnt_acc(num_options):
    d = {}
    for i in range(num_options):
        d[chr(ord('a') + i)] = {'correct': 0, 'total': 0}
    return d

def cal_acc(inf, metrics):
    num_correct = len([d for d in inf if d['correct']])
    metrics['acc'] = num_correct / len(inf)
    counter = {'2_human': init_cnt_acc(2), '2_animal': init_cnt_acc(2), '4': init_cnt_acc(4)}
    for d in inf:
        if len(d['options']) == 4:
            type_option = '4'
        elif d['image'].startswith('human_'):
            type_option = '2_human'
        else:
            type_option = '2_animal'
        if d['correct']:
            counter[type_option][d['gold'].lower()]['correct'] += 1
        counter[type_option][d['gold'].lower()]['total'] += 1
    for type_option in counter:
        num_options = int(type_option.split('_')[0])
        for i in range(num_options):
            counter[type_option][chr(ord('a') + i)] = counter[type_option][chr(ord('a') + i)]['correct'] / counter[type_option][chr(ord('a') + i)]['total']
    metrics['acc_per_option'] = counter

def merge_metrics(dicts, ndigits=4):
    def recursive_avg(ds):
        result = {}
        for k in ds[0].keys():
            values = [d[k] for d in ds]
            if isinstance(values[0], dict):
                result[k] = recursive_avg(values)
            else:
                avg = sum(values) / len(values)
                result[k] = round(avg, ndigits)
        return result
    return recursive_avg(dicts)

def eval_multirun(model, num_run=3):
    files = sorted(os.listdir(INFERENCE_DIR))
    model_metrics = []
    if num_run == 1:
        file = f"{model}.jsonl"
        assert file in files
        inf = read_jsonl(os.path.join(INFERENCE_DIR, file))
        judge(inf)
        metrics = cal_prf(inf)
        cal_acc(inf, metrics)
        return metrics
    else:
        for idx in range(num_run):
            file = f"{model}_run{idx}.jsonl"
            assert file in files
            inf = read_jsonl(os.path.join(INFERENCE_DIR, file))
            judge(inf)
            metrics = cal_prf(inf)
            cal_acc(inf, metrics)
            model_metrics.append(metrics)
        return merge_metrics(model_metrics)

if __name__ == '__main__':
    results = {}
    with open(ORDER_PATH, 'r') as f:
        models = [item.strip() for item in f.readlines() if item]
    for model in models:
        results[model] = eval_multirun(model)
        # results[model] = eval_multirun(model, 1)
    with open(DST_PATH, 'w') as f:
        json.dump(results, f, indent=2)