import os
import numpy as np
import torch
import json
from .tokenizers import TOKENIZERS

from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map


class DeepSeekDistillation(Dataset):
    def __init__(self, dataset_dir, max_seq_len, tokenizer, padding = True) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.files = sorted([dir for dir in os.listdir(dataset_dir) if dir.endswith('.txt')]) # 3414 * 6400 = 21,849,600 files train, test 60065 lines
        self._load_file(0)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.combine_sequence = int(np.ceil(max_seq_len / 1024))
        self.lines_per_file = len(self.file_texts) // self.combine_sequence
        self.padding = padding

    def __len__(self):
        return len(self.files) * self.lines_per_file

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(f'{self.dataset_dir}/{self.files[self.cur_file_idx]}') as f:
            self.file_texts = f.readlines()

    def __getitem__(self, index):
        file_idx = index // self.lines_per_file
        if file_idx != self.cur_file_idx:
            self._load_file(file_idx)
        line_idx = index % self.lines_per_file
        if line_idx > len(self.file_texts):
            print(f'{index}, {line_idx}, {len(self.file_texts)}, {self.dataset_dir}/{self.files[file_idx]}')
            line_idx = len(self.file_texts) - 1

        text = self.file_texts[line_idx * self.combine_sequence]
        for i in range(line_idx * self.combine_sequence + 1, line_idx * self.combine_sequence + self.combine_sequence):
            text += self.file_texts[i]

        if self.padding:
            token_ids = self.tokenizer(text, padding='max_length', 
                                   max_length = self.max_seq_len + 1, padding_side='right', 
                                   truncation=True, return_tensors='pt').input_ids
            real_len = len(self.tokenizer(text).input_ids)
        else:
            token_ids = self.tokenizer(text, return_tensors='pt').input_ids
            real_len = len(token_ids[0])
        return token_ids[0, :-1], token_ids[0, 1:], real_len


class Tokenized_data(Dataset):
    def __init__(self, tokenizer, is_test = False, max_total = -1, start_from = 0) -> None:
        super().__init__()
        # self.folder_prefix = '/home/fdong/data/openweb_every_6400' if is_test else '/home/fdong/data/openweb_every_6400'
        self.folder_prefix = '/home/fdong/data/legal/train'
        self.domain_files = os.listdir(self.folder_prefix) # 3414 * 6400 = 21,849,600 files train, test 60065 lines
        self.domain_files = sorted(self.domain_files)
        self.cur_file_idx = 0
        if is_test:
            with open(f'{self.folder_prefix}/{self.domain_files[-1]}') as f:
                self.file_texts = f.readlines()
        else:
            with open(f'{self.folder_prefix}/{self.domain_files[0]}') as f:
                self.file_texts = f.readlines()
        self.tokenizer = tokenizer
        self.window_size = 512
        self.is_test = is_test
        self.start_from = start_from

    def __len__(self):
        return 2048 if self.is_test else len(self.domain_files) * 6400 - self.start_from 

    def __getitem__(self, index):
        if self.is_test:
            line_idx = index
        else:
            index += self.start_from
            file_idx = index // 6400
            if file_idx != self.cur_file_idx:
                with open(f'{self.folder_prefix}/{self.domain_files[file_idx]}') as f:
                    self.file_texts = f.readlines()
                self.cur_file_idx = file_idx
            line_idx = index % 6400
            if line_idx > len(self.file_texts):
                print(f'{index}, {line_idx}, {len(self.file_texts)}, {self.folder_prefix}/{self.domain_files[file_idx]}')
        text = self.file_texts[line_idx].replace('<|', '').replace('|>', '')

        tokens = self.tokenizer(text, return_tensors='pt').input_ids
        source, target = tokens[0, :-1], tokens[0,1:]
        return source, target, 0
