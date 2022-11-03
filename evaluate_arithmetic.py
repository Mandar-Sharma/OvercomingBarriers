import torch
import pickle
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from transformers import BertTokenizerFast, BertForMaskedLM, DistilBertTokenizerFast, DistilBertForMaskedLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForMaskedLM.from_pretrained('./Models/Arith/checkpoint-54000')

dataset = load_dataset("./Datasets/Arith")

def encode_with_truncation(examples):
    ab = [example.split('=')[0] + "= [MASK] [MASK] [MASK]" for example in examples["text"]]
    return tokenizer(ab, truncation=True, padding="max_length", max_length=512, return_special_tokens_mask=True)

test_dataset = dataset['test'].map(encode_with_truncation, batched=True)
test_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'special_tokens_mask'])

model.eval()
ground = []
preds = []

for idx, val in tqdm(enumerate(dataset['test'])):
    out = model(input_ids = test_dataset[idx]['input_ids'].reshape(1, 512), attention_mask = test_dataset[idx]['attention_mask'].reshape(1, 512))
    mask_idx = (test_dataset[idx]['input_ids'] == 103).nonzero(as_tuple=True)[0]
    label_ids = tokenizer.encode(val['text'].split('=')[1][1:])[1:-1]
    preds_ids = []
    for i,v in enumerate(label_ids):
        preds_ids.append(torch.argmax(out['logits'][0][mask_idx[i]]))
    try:
        preds.append(int(tokenizer.decode(preds_ids)))
        ground.append(int(val['text'].split('=')[1][1:]))
    except ValueError as e:
        pass
    
print(mean_squared_error(ground, preds, squared = True))
print(mean_squared_error(ground, preds, squared = False))
print(mean_squared_log_error(ground, preds, squared = True))
print(mean_squared_log_error(ground, preds, squared = False))