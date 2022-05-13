# developed based on source file: 
# method/prompttune/tasks/prompt_tune_nli.py
# Author: Chang Xue
# Date: 2022/05/12

import torch


from panguapp.megatron import get_args
from panguapp.megatron import print_rank_0
from panguapp.megatron import get_timers
from panguapp.megatron import get_tokenizer
from panguapp.megatron import mpu
from panguapp.megatron.utils import get_ltor_masks_and_position_ids
from panguapp.megatron.utils import reduce_losses

from panguapp.method.prompttune.datadefine.dataset import BiliDataset
from panguapp.method.prompttune.model.gpt2_model_pt import GPT2ModelPT
from panguapp.method.prompttune.tasks.prompt_tune import prompt_tune_train
from panguapp.method.prompttune.paras.custom_arguments import add_custom_args


def model_provider():
    print_rank_0('building PT model ...')
    model = GPT2ModelPT(num_tokentypes=0, parallel_output=True)
    return model


def get_batch(data_iterator):
    args = get_args()
    tokenizer = get_tokenizer()

    if data_iterator:
        data = next(data_iterator)
    else:
        data = None

    data_b = mpu.broadcast_data(['input_ids', 'label_ids', 'context_lengths', 'loss_mask'], data, torch.int64)

    tokens_ = data_b['input_ids'].long()
    target = data_b['label_ids'].long()
    context_lengths = data_b['context_lengths'].long()
    loss_mask = data_b['loss_mask'].float()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    loss_mask = loss_mask[:, :-1].contiguous()
    context_lengths[context_lengths >= args.seq_length] = args.seq_length - 1

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(tokens, tokenizer.eod, args.reset_position_ids,
                                                                      args.reset_attention_mask, args.eod_mask_loss)

    return tokens, labels, target, loss_mask, attention_mask, position_ids, context_lengths


def get_preds(logits, loss_mask):
    tokenizer = get_tokenizer()
    cand_ids = BiliDataset.get_cand_ids(tokenizer)

    scores = (torch.sum(logits * loss_mask.unsqueeze(-1), 1) / torch.sum(loss_mask, -1).unsqueeze(-1))[:, cand_ids]
    preds = torch.argmax(scores, dim=-1)

    return preds


def forward_step(data_iterator, model, parallel_output=None):
    timers = get_timers()

    timers('batch generator').start()
    tokens, labels, target, loss_mask, attention_mask, position_ids, context_lengths = get_batch(data_iterator)
    timers('batch generator').stop()

    loss_, logits = model(tokens, position_ids, attention_mask, labels=labels, forward_method_parallel_output=parallel_output)
    loss = torch.sum(loss_ * loss_mask) / loss_mask.sum()

    preds = get_preds(logits, loss_mask)

    return loss, {'lm loss': reduce_losses([loss])[0]}, target, preds


def train_valid_test_datasets_provider():
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0('> building train, validation, and test datasets for pangu ...')

    train_ds = BiliDataset(args, tokenizer, "train.json")
    valid_ds = BiliDataset(args, tokenizer, "dev.json")
    test_ds = BiliDataset(args, tokenizer, "dev.json")
    print_rank_0("> finished creating pangu finetune datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    prompt_tune_train(train_valid_test_datasets_provider, model_provider, forward_step, extra_args_provider=add_custom_args, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
