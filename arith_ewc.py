import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import pickle
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForMaskedLM, DistilBertTokenizerFast, DistilBertForMaskedLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

with open('./fisher.pkl','rb') as fp:
    sum_sqr_gradients = pickle.load(fp)
    
p = 0
for g in sum_sqr_gradients:
    if len(g.shape) == 1:
        p += g.shape[0]
    else:
        p += (g.shape[0] * g.shape[1])
print(p)

_buff_param_names = [param[0].replace('.', '__') for param in model.named_parameters()]
for _buff_param_name, param in tqdm(zip(_buff_param_names, sum_sqr_gradients)):
    model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone())
    
for param_name, param in tqdm(model.named_parameters()):
    _buff_param_name = param_name.replace('.', '__')
    model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())
    
dataset = load_dataset("./Datasets/Arith")

def encode_with_truncation(examples):
    ab = [example.split('=')[0] + "= [MASK] [MASK] [MASK]" for example in examples["text"]]
    return tokenizer(ab, truncation=True, padding="max_length", max_length=512, return_special_tokens_mask=True)

# tokenizing the train dataset
train_dataset = dataset['train'].map(encode_with_truncation, batched=True)
train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'special_tokens_mask'])

label_col = []
for idx, val in tqdm(enumerate(dataset['train'])):
    label_ids = tokenizer.encode(val['text'].split('=')[1][1:])[1:-1]
    mask_idx = (train_dataset[idx]['input_ids'] == 103).nonzero(as_tuple=True)[0]
    #labels = torch.full([512],-100)
    labels = [-100] * 512
    for i, v in enumerate(label_ids):
        labels[mask_idx[i]] = v
    label_col.append(labels)

train_dataset = train_dataset.add_column("labels", label_col)

training_args = TrainingArguments(
    output_dir='./Models/ArithEWC',
    overwrite_output_dir=True,      
    num_train_epochs=50,           
    per_device_train_batch_size=16,
    logging_steps=500,           
    save_steps=1000,
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        losses = []
        for param_name, param in model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            estimated_mean = getattr(model, '{}_estimated_mean'.format(_buff_param_name))
            estimated_fisher = getattr(model, '{}_estimated_fisher'.format(_buff_param_name))
            losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
        loss_ewc = (p * 1e-8 / 2) * sum(losses)
        loss += loss_ewc
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)


trainer.train()