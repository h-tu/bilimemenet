# developed based on source file: 
# model/pangu_evolution/inference/infer_generate.py
# Author: Chang Xue
# Date: 2022/05/12

import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP


from panguapp.megatron import get_args, mpu
from panguapp.megatron import get_tokenizer
from panguapp.megatron.checkpointing import load_checkpoint
from panguapp.megatron.initialize import initialize_panguapp.megatron
from panguapp.megatron.fp16 import FP16_Module
from panguapp.megatron.model import DistributedDataParallel as LocalDDP
from panguapp.megatron.utils import get_ltor_masks_and_position_ids
from panguapp.model.pangu_evolution.model.model_define import PanguModelEnhance

class InferGenerate(object):

    def __init__(self):
        initialize_panguapp.megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

        self.args = get_args()
        self.tokenizer = get_tokenizer()

        self.model = self.get_model()
        load_checkpoint(self.model, None, None)
        self.model.eval()

    def top_k_logits(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        if top_k > 0:
            mask1 = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[mask1] = filter_value

        batch_size = logits.size()[0]
        if top_p > 0.0:
            logits = logits.view(batch_size, -1).contiguous()
            for i in logits:
                logits_temp, indices_temp = torch.sort(i, descending=True)
                probs = torch.cumsum(F.softmax(logits_temp, dim=-1), dim=-1)
                sorted_mask = probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0
                mask2 = indices_temp[sorted_mask]
                i[mask2] = filter_value

            logits = logits.view(batch_size, -1).contiguous()

        return logits

    def get_model(self):
        args = get_args()
        model = PanguModelEnhance(num_tokentypes=0, parallel_output=False)
        if mpu.get_data_parallel_rank() == 0:
            rank_id = mpu.get_model_parallel_rank()
            temp = []
            for p in model.parameters():
                temp.append(p.nelement())
            parameter_num = sum(temp)
            print(f" > number of parameters on model parallel rank {rank_id}: {parameter_num}.", flush=True)

        model.cuda(torch.cuda.current_device())

        if args.fp16:
            model = FP16_Module(model)

        if args.DDP_impl == 'torch':
            i = torch.cuda.current_device()
            model = torchDDP(model, device_ids=[i], output_device=i, process_group=mpu.get_data_parallel_group())
            return model
        elif args.DDP_impl == 'local':
            model = LocalDDP(model)
            return model

        raise NotImplementedError('Unknown DDP implementation specified: {}. Exiting.'.format(args.DDP_impl))

    def get_single_generate_data(self, input_str, max_len):
        ids = self.tokenizer.tokenize(input_str)
        ids = np.array(ids).reshape(1, -1)
        if ids.shape[-1] >= max_len:
            ids = ids[:, -max_len:]

        bs, l = ids.shape
        context_len = l
        pad_length = self.args.seq_length - l
        input = np.pad(ids, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, self.tokenizer.pad_id))

        input = torch.tensor(input, dtype=torch.long).contiguous().cuda()
        mask, temp, position = get_ltor_masks_and_position_ids(input, self.tokenizer.eod, self.args.reset_position_ids, self.args.reset_attention_mask, self.args.eod_mask_loss)
        input_data = (input, position, mask, l)
        return input_data, context_len

    def do_single_generate(self, input_data, temperature=1, top_k=None, top_p=None, max_num=50):
        (input_ids, position_ids, attention_mask, valid_length) = input_data
        inputs = input_ids, None

        cnt = 0
        with torch.no_grad():
            while valid_length < self.args.seq_length and cnt > max_num:
                logits = self.model(inputs, position_ids, attention_mask)

                n_logits = logits[:, valid_length - 1, :] / temperature

                if not top_k and not top_p:
                    token = torch.argmax(n_logits, dim=-1)
                else:
                    logscores = self.top_k_logits(n_logits, top_k=top_k, top_p=top_p)
                    probs = F.softmax(logscores, dim=-1)
                    token = torch.multinomial(probs.float(), num_samples=1).squeeze(1)

                if token[0] == self.tokenizer.eod or valid_length == self.args.seq_length - 1 or cnt >= max_num:
                    outputs = input_ids
                    break

                input_ids[:, valid_length] = token
                valid_length += 1
                cnt += 1

        return outputs

    def generate(self, input_str, temperature=1, top_k=None, top_p=None, max_len=1000, max_num=50):
        input_data, context_len = self.get_single_generate_data(input_str, max_len)

        outputs = self.do_single_generate(input_data, temperature=temperature, top_k=top_k, top_p=top_p, max_num=max_num)
        outputs = outputs.cpu().numpy()
        length = np.sum(outputs != self.tokenizer.pad_id)
        outputs = outputs[0, context_len:length]

        generate_text = "".join(self.tokenizer.convert_ids_to_tokens(outputs.tolist()))
        return generate_text
