# from transformers import T5ForConditionalGeneration
# import torch

# model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

# input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + 3  # add 3 for special tokens
# labels = (
#     torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + 3
# )  # add 3 for special tokens

# loss = model(input_ids, labels=labels).loss  # forward pass

# print(loss)

#pytorch_lightning==0.7.5

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


train_pos_files = glob.glob( data_path +'/train/pos/*.txt')
train_neg_files = glob.glob( data_path +'/train/neg/*.txt')

random.shuffle(train_pos_files)
random.shuffle(train_neg_files)



## move val data
# val_pos_files = train_pos_files[:1000]
# val_neg_files = train_neg_files[:1000]
# for f in val_pos_files:
#     shutil.move(f, data_path+'/val/pos')
# for f in val_neg_files:
#     shutil.move(f,  data_path+'/val/neg')

val_pos_files = glob.glob( data_path +'/val/pos/*.txt')
val_neg_files = glob.glob( data_path +'/val/neg/*.txt')
print('val:')
print(len(val_neg_files),len(val_pos_files))

tokenizer = T5Tokenizer.from_pretrained('t5-base')

ids_neg = tokenizer.encode('negative </s>')
ids_pos = tokenizer.encode('positive </s>')

dataset = ImdbDataset(tokenizer, data_path, 'val',  max_len=512)
print('dataset:')
print(len(dataset))

args_dict.update({'output_dir': 't5_imdb_sentiment', 'num_train_epochs':2})
args = argparse.Namespace(**args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)

def get_dataset(tokenizer, type_path, args):
  return ImdbDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)

model = T5FineTuner(args)
print(model.val_dataloader())

trainer = pl.Trainer(**train_params)
trainer.fit(model)

model.model.save_pretrained('t5_base_imdb_sentiment')


#####################################################
# Evaluation

# import textwrap
# from tqdm.auto import tqdm
# from sklearn import metrics

# dataset = ImdbDataset(tokenizer, data_path, 'test',  max_len=512)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)

# it = iter(loader)
# batch = next(it)
# batch["source_ids"].shape

# outs = model.model.generate(input_ids=batch['source_ids'].cuda(), attention_mask=batch['source_mask'].cuda(), max_length=2)

# dec = [tokenizer.decode(ids) for ids in outs]
# texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
# targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

