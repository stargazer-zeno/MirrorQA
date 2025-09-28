from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT
import os
import json
from tqdm import tqdm

model_path = "xxx"
RESULT_DIR = RESULT_DIR['base']
os.makedirs(RESULT_DIR, exist_ok=True)
run_idx = 0
result_file = f"instructblip_run{run_idx}.jsonl"

model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
processor = InstructBlipProcessor.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def chat(entry):
    question = entry['question']
    image_name = entry['image']
    options = entry['options']
    image_filepath = os.path.join(IMAGE_DIR, image_name)
    
    image = Image.open(image_filepath).convert("RGB")
    prompt = ROLE_PROMPT + f"Input: Question: {question}, Options: {'; '.join(options)}.\nOutput:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=20
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()[len(prompt):].strip()
    return generated_text

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