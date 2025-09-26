import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT
import os
import json
from tqdm import tqdm

model_path = "xxx"
RESULT_DIR = RESULT_DIR['base']
os.makedirs(RESULT_DIR, exist_ok=True)
run_idx = 0
result_file = f"minicpm_run{run_idx}.jsonl"

model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def chat(entry):
    question = entry['question']
    image_name = entry['image']
    options = entry['options']
    image_filepath = os.path.join(IMAGE_DIR, image_name)
    
    image = Image.open(image_filepath).convert("RGB")
    prompt1 = ROLE_PROMPT + "Input: Image: "
    prompt2 = f"\nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"

    msgs = [{'role': 'user', 'content': [prompt1, image, prompt2]}]

    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        max_new_tokens=20,
        do_sample=False
    )
    return res.strip()

with open(DATA_PATH, 'r', encoding="utf-8") as f, open(os.path.join(RESULT_DIR, result_file), 'w+', encoding="utf-8") as fout:
    lines = f.readlines()
    num_lines = len(lines)
    for line in tqdm(lines, total=num_lines, desc="Processing entries"):
        entry = json.loads(line)
        pred = chat(entry)
        if len(pred) == 0:
            pred = '--'
        gold_patterns = []
        result_json = {"image": entry['image'], "question": entry['question'], "options": entry['options'], "pred": pred, "gold": entry['answer']}
        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')