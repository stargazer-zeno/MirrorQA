from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT
import os
import json
from tqdm import tqdm

model_path = "xxx"
RESULT_DIR = RESULT_DIR['base']
os.makedirs(RESULT_DIR, exist_ok=True)
run_idx = 0
result_file = f"mplug_run{run_idx}.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
processor = model.init_processor(tokenizer)

def chat(entry):
    question = entry['question']
    image_name = entry['image']
    options = entry['options']
    image_filepath = os.path.join(IMAGE_DIR, image_name)
    
    image = Image.open(image_filepath).convert("RGB")

    prompt = ROLE_PROMPT + f"Input: Image: <|image|>\nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""}
    ]

    inputs = processor(messages, images=[image], videos=None)

    inputs.to(device)
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens':20,
        'decode_text':True,
        'do_sample': False
    })

    g = model.generate(**inputs)
    return g[0].strip()

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