import os
import random
import base64
import jsonlines
import json
from openai import OpenAI

TEST_DATA_PATH = "xxx"
TRAIN_DATA_PATH = "xxx"
IMAGE_DIR = "xxx"
RESULT_DIR = "xxx"
ICL_PATH = "xxx"

client = OpenAI(
    base_url="xxx",
    api_key="xxx",
)

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data
train_data = read_jsonl(TRAIN_DATA_PATH)

def encode_image_to_base64(path):
  with open(path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def init_json(file_path):
    with open(file_path, 'w') as f:
        json.dump({}, f, indent=2)

def construct_few_shot(entry, num_shot, model, mode="random"):
    global train_data

    icl_file = os.path.join(ICL_PATH, f'{model}_{num_shot}shot.json')
    if not os.path.exists(icl_file):
        init_json(icl_file)
    with open(icl_file, 'r') as f:
        icl_dict = json.load(f)

    img_path = os.path.join(IMAGE_DIR, entry['image'])
    base64_image = encode_image_to_base64(img_path)
    role_prompt = "You are currently a senior expert in visual reasoning.\nGiven an Image, a Question, and Options, your task is to choose the correct answer.\nNote that you only need to choose one option from all options without explaining any reason.\n"
    test_content = [
        {"type": "text", "text": "Input: Image: "},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        {"type": "text", "text": f"\nQuestion: {entry['question']}, Options: {'; '.join(entry['options'])}.\nOutput:"}
    ]
    
    fewshot_prompt = f"Given the following {num_shot} examples to learn the visual reasoning task:\n" if num_shot else ""
    content = [
        {"type": "text", "text": role_prompt + fewshot_prompt},
    ]

    if mode == "random":
        if entry['image'] in icl_dict:
            sampled_data = icl_dict[entry['image']]
        else:
            sampled_data = random.sample(train_data, num_shot)
            icl_dict[entry['image']] = sampled_data
    else:
        sampled_data = []
    
    with open(icl_file, 'w') as f:
        json.dump(icl_dict, f, indent=2)

    if len(sampled_data) != num_shot:
        raise ValueError("len(sampled_data) != num_shot")
    else:
        fewshot_content = []
        for idx, example in enumerate(sampled_data):
            ex_img_path = os.path.join(IMAGE_DIR, example['image'])
            ex_base64_image = encode_image_to_base64(ex_img_path)
            ex_answer_str = example['options'][ord(example['answer']) - ord('A')]
            fewshot_content.extend([
                {"type": "text", "text": f"Example{idx+1}:\nInput: Image: "},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_base64_image}"}},
                {"type": "text", "text": f"\nQuestion: {example['question']}, Options: {'; '.join(example['options'])}.\nOutput: {ex_answer_str}\n"}
            ])
    content.extend(fewshot_content)
    content.extend(test_content)
    return content