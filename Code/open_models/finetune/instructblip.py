import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

MODEL_PATH = "xxx"
TRAIN_DATA_PATH = "xxx"
OUTPUT_DIR = "xxx"

class InstructBlipDataset(Dataset):
    def __init__(self, jsonl_path, processor, max_length=512):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_path = entry['images'][0]
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            return None

        prompt = entry['messages'][0]['content']
        answer = entry['messages'][1]['content']
        
        # 预处理文本和图像
        encoding = self.processor(
            images=image, 
            text=prompt, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 预处理标签
        labels = self.processor.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=20
        ).input_ids
        
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        encoding["labels"] = labels.squeeze()
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding

processor = InstructBlipProcessor.from_pretrained(MODEL_PATH)
model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

train_dataset = InstructBlipDataset(
    jsonl_path=TRAIN_DATA_PATH,
    processor=processor,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_train=True,
    num_train_epochs=60,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    
    logging_steps=10,
    save_strategy="steps",
    save_steps=10,
    
    bf16=True,
    remove_unused_columns=False,
    dataloader_num_workers=16,
    
    load_best_model_at_end=False,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"model saved: {OUTPUT_DIR}")