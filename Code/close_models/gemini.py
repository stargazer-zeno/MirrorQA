import os
from util import TEST_DATA_PATH, RESULT_DIR, construct_few_shot, client
import os
import json
from tqdm import tqdm

os.makedirs(RESULT_DIR, exist_ok=True)
num_shot = 0
result_file = f"gemini_{num_shot}shot.jsonl"

def chat(entry, num_shot=0):
    messages = [{"role": "user", "content": construct_few_shot(entry, num_shot, 'gemini')}]
    completion = client.chat.completions.create(
        model="google/gemini-2.5-pro",
        messages=messages,
        temperature=0,
        seed=42
    )
    return completion.choices[0].message.content

with open(TEST_DATA_PATH, 'r', encoding="utf-8") as f, open(os.path.join(RESULT_DIR, result_file), 'w+', encoding="utf-8") as fout:
    lines = f.readlines()
    num_lines = len(lines)
    for line in tqdm(lines, total=num_lines, desc="Processing entries"):
        entry = json.loads(line)
        pred = chat(entry, num_shot)
        if len(pred) == 0:
            pred = '--'
        gold_patterns = []
        result_json = {"image": entry['image'], "question": entry['question'], "options": entry['options'], "pred": pred, "gold": entry['answer']}
        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')