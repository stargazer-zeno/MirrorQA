import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT
import os
import json
from tqdm import tqdm

model_path = "xxx"
RESULT_DIR = RESULT_DIR['base']
os.makedirs(RESULT_DIR, exist_ok=True)
run_idx = 0
result_file = f"llama_run{run_idx}.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
processor = AutoProcessor.from_pretrained(model_path)

def chat(entry):
    question = entry['question']
    image_name = entry['image']
    options = entry['options']
    image_filepath = os.path.join(IMAGE_DIR, image_name)
    
    image = Image.open(image_filepath).convert("RGB")

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": ROLE_PROMPT+"Input: Image: "},
            {"type": "image"},
            {"type": "text", "text": f"\nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    return processor.decode(output[0], skip_special_tokens=True).split("Output:assistant")[-1].strip()

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