# developed based on source file: 
# method/prompttune/datadefine/dataset.py
# Author: Chang Xue
# Date: 2022/05/12

import os
import json
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, args, tokenizer, split, ratio=1, num=-1):
        self.args = args
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.split = split

        self.data, self.max_len = self.process_data()
        if num > 0:
            self.data = self.data[:num]

    def collate(self, samples):
        bs = len(samples)

        model_data = {
            "input_ids": torch.ones(bs, self.args.seq_length, dtype=torch.long) * self.tokenizer.pad_id,
            "label_ids": torch.ones(bs, 1, dtype=torch.long),
            "context_lengths": torch.ones(bs, 1, dtype=torch.long),
            "loss_mask": torch.zeros(bs, self.args.seq_length, dtype=torch.long)
        }

        for i, samp in enumerate(samples):
            seq_len = len(samp["input_ids"])
            model_data["input_ids"][i][:seq_len] = torch.tensor(samp["input_ids"], dtype=torch.long)
            model_data["label_ids"][i][0] = samp["label_ids"]
            model_data["context_lengths"][i][0] = seq_len
            model_data["loss_mask"][i, seq_len-2] = 1

        return model_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BiliDataset(BaseDataset):

    LABELS = {'动画': 0, '音乐舞蹈': 1, '知识': 2, '生活': 3, '影视': 4}
    template = "网络用语：{sentence1}\n选项：{sentence2}\n答案：{label}"

    def __init__(self, args, tokenizer, split, ratio=1, num=-1):
        super().__init__(args, tokenizer, split, ratio, num)

    @staticmethod
    def get_cand_ids(tokenizer):
        cand_ids = []
        if not cand_ids:
            for i in range(1, 4):
                cand_ids.append(tokenizer.encode(str(i)))
        else:
            for cand_id in cand_ids:
                cand_ids.append(cand_id[0])
        return cand_ids

    def process_data(self):
        data = []
        sizes = []
        data_path = os.path.join(self.args.data_path, self.split)
        with open(data_path, "r", encoding="utf-8") as f:
            data_lines = f.readlines()
        print(f"All {self.split} case num: {len(data_lines)}.")

        for i, instance in enumerate(data_lines):
            instance = json.loads(instance)

            sentence1 = instance["sentence1"]
            sentence2 = instance["sentence2"]
            label = instance.get("label", "-100")

            if label not in self.LABELS:
                continue

            label_id = self.LABELS.get[label]
            input_id_str = self.template.replace("{sentence1}", sentence1)
            input_id_str = input_id_str.replace("{sentence2}", sentence2)
            input_id_str = input_id_str.replace("{label}", str(label_id+1))
            input_id = self.tokenizer.encode(input_id_str)

            sizes.append(len(input_id))

            data.append({
                "idx": i,
                "input_ids": input_id,
                "label_ids": label_id,
            })

        max_len = max(sizes)

        return data, max_len