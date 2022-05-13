import textwrap
from tqdm.auto import tqdm
from sklearn import metrics

import argparse
from T5Finetunner import T5FineTuner, LoggingCallback
import glob
import random
import shutil
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from T5Finetunner import ImdbDataset 

data_path = "/media/zihao/New Volume1/UMASS/685_e/github/Zihao_branch/data/aclImdb_v1/aclImdb"

args_dict = dict(
    data_dir= data_path, # path for data files
    output_dir="checkpoints", # path to save the checkpoints
    model_name_or_path='t5-base',
    tokenizer_name_or_path='t5-base',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=1,
    eval_batch_size=1,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)
args = argparse.Namespace(**args_dict)

tokenizer = T5Tokenizer.from_pretrained('t5-base')

dataset = ImdbDataset(tokenizer, data_path, 'test',  max_len=512)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

it = iter(loader)

batch = next(it)
batch["source_ids"].shape

device = torch.device("cuda")
model = T5FineTuner(args)
model = T5ForConditionalGeneration.from_pretrained('t5_base_imdb_sentiment')
# ckpt = torch.load('t5_imdb_sentiment/checkpointepoch=1.ckpt')
# #model = model.load_state_dict(torch.load("t5_base_imdb_sentiment/pytorch_model.bin"))
# model.load(ckpt['state_dict'])
model.to(device)


#Single test
############################################################
# outputs = []
# targets = []
# outs = model.generate(input_ids=batch['source_ids'].cuda(), 
#                               attention_mask=batch['source_mask'].cuda(), 
#                               max_length=2)

# dec = [tokenizer.decode(ids) for ids in outs]

# texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
# targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

# for i,t in enumerate(targets):
#     targets[i] = t.replace('</s>','')
# for i,t in enumerate(outputs):
#     outputs[i] = t.replace('<pad> ','')
##################################################################


# for i in range(32):
#     lines = textwrap.wrap("Review:\n%s\n" % texts[i], width=100)
#     print("\n".join(lines))
#     print("\nActual sentiment: %s" % targets[i])
#     print("Predicted sentiment: %s" % dec[i])
#     print("=====================================================================\n")
    
loader = DataLoader(dataset, batch_size=32, num_workers=4)
model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
  outs = model.generate(input_ids=batch['source_ids'].cuda(), 
                              attention_mask=batch['source_mask'].cuda(), 
                              max_length=2)

  dec = [tokenizer.decode(ids) for ids in outs]
  target = [tokenizer.decode(ids) for ids in batch["target_ids"]]
  
  for i,t in enumerate(target):
    target[i] = t.replace('</s>','')
  for i,t in enumerate(dec):
    dec[i] = t.replace('<pad> ','')
  
  
  outputs.extend(dec)
  targets.extend(target)
  
print(metrics.accuracy_score(targets, outputs))
print(metrics.classification_report(targets, outputs))