import torch
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling

wikiset = load_dataset("wikipedia", "20220301.simple")

tokenizer = BertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = BertForMaskedLM.from_pretrained('distilbert-base-uncased')

def encode_with_truncation(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = wikiset['train'].map(encode_with_truncation, batched=True)
train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask'])

device = torch.device("cuda:0")
model.cuda()
model = model.to(device)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

ssg = []
sqr_gradients = []
sum_sqr_gradients = []

file_idx = 0

for i in tqdm(train_dataset):
    batch = data_collator([i])
    batch.to(device)
    out = model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'], labels=batch['labels'])
    grads = torch.autograd.grad(out['loss'], model.parameters())
    sqr_gradients.append([x.cpu() ** 2 for x in grads])
    if len(sqr_gradients) == 100:
        for i in range(len(sqr_gradients[0])):
            items = [item[i] for item in sqr_gradients]
            sum_sqr_gradients.append(torch.mul(torch.sum(torch.stack(items),0), 1/205328))
        ssg.append(sum_sqr_gradients)
        del sqr_gradients
        del items
        del sum_sqr_gradients
        sqr_gradients = []
        sum_sqr_gradients = []
    if len(ssg) == 100:
        with open('./Gradients/DistilBERT/Wiki/ssg_{}.pkl'.format(file_idx), 'wb') as fp:
            pickle.dump(ssg, fp)
        del ssg
        ssg = []
        file_idx += 1
    del batch
    del out
    del grads
    torch.cuda.empty_cache()
    
with open('./Gradients/DistilBERT/Wiki/ssg_{}.pkl'.format(file_idx),'wb') as fp:
    pickle.dump(ssg, fp)
    
files = glob('./Gradients/DistilBERT/Wiki/ssg_*')
fischer_1 = []
ssgs = []
for file in tqdm(files):
    with open(file, 'rb') as fp:
        ssg = pickle.load(fp)
    for i in range(len(ssg[0])):
        items = [item[i] for item in ssg]
        ssgs.append(torch.sum(torch.stack(items),0))
    fischer_1.append(ssgs)
    del ssg
    del ssgs
    ssgs = []
    
fischer_2 = []
for i in tqdm(range(len(fischer_1[0]))):
    items = [item[i] for item in fischer_1]
    fischer_2.append(torch.sum(torch.stack(items),0))
    
with open('./fisher.pkl','wb') as fp:
    pickle.dump(fischer_2, fp)

with open('./Gradients/DistilBERT/Wiki/fisher.pkl','wb') as fp:
    pickle.dump(fischer_2, fp)