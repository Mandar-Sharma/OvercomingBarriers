import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import pickle
from tqdm import tqdm
from datasets import load_dataset, load_metric
from transformers import BertTokenizer, BertForMaskedLM, DistilBertTokenizerFast, DistilBertForMaskedLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

dataset = load_dataset("./Datasets/Arith")

def encode_with_truncation(examples):
    ab = [example.split('=')[0] + "= [MASK] [MASK] [MASK]" for example in examples["text"]]
    return tokenizer(ab, truncation=True, padding="max_length", max_length=512, return_special_tokens_mask=True)

train_dataset = dataset['train'].map(encode_with_truncation, batched=True)
train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'special_tokens_mask'])

label_col = []
for idx, val in tqdm(enumerate(dataset['train'])):
    label_ids = tokenizer.encode(val['text'].split('=')[1][1:])[1:-1]
    mask_idx = (train_dataset[idx]['input_ids'] == 103).nonzero(as_tuple=True)[0]
    labels = [-100] * 512
    for i, v in enumerate(label_ids):
        labels[mask_idx[i]] = v
    label_col.append(labels)

train_dataset = train_dataset.add_column("labels", label_col)

training_args = TrainingArguments(
    output_dir='./Models/Arith1',
    overwrite_output_dir=True,      
    num_train_epochs=50,           
    per_device_train_batch_size=16,
    logging_steps=500,           
    save_steps=1000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()