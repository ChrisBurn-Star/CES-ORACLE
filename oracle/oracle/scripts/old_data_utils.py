import os
import torch
from torch.utils.data import Dataset
import random
import string
import pyarrow.dataset as ds
from datasets import load_from_disk
import numpy as np
class FakeData(Dataset):
    def __init__(self, window_size, vocab_size) -> None:
        super().__init__()
        self.window_size = window_size
        self.vocab_size = vocab_size

    def __len__(self):
        return 100000

    def __getitem__(self, index):
        return torch.rand([self.window_size]).long() + 10, torch.rand([self.window_size]).long() + 10, 0


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
        self.window_size = 2048
        self.is_test = is_test
        self.start_from = start_from

    def __len__(self):
        return 2048 if self.is_test else len(self.domain_files) * 4096 - self.start_from 

    def __getitem__(self, index):
        if self.is_test:
            line_idx = index
        else:
            index += self.start_from
            file_idx = index // 4096
            if file_idx != self.cur_file_idx:
                with open(f'{self.folder_prefix}/{self.domain_files[file_idx]}') as f:
                    self.file_texts = f.readlines()
                self.cur_file_idx = file_idx
            line_idx = index % 4096
            if line_idx > len(self.file_texts):
                print(f'{index}, {line_idx}, {len(self.file_texts)}, {self.folder_prefix}/{self.domain_files[file_idx]}')

        text = self.file_texts[line_idx].replace('<|', '').replace('|>', '')

        tokens = self.tokenizer(text, padding='max_length', 
                                   max_length = self.window_size + 1, padding_side='right', 
                                   truncation=True, return_tensors='pt').input_ids
        source, target = tokens[0, :-1], tokens[0,1:]
        return source


        return ids
    def replace_with_punctuation(self, text, num_replacements=None):
        """随机将文本中的字符替换为标点，并返回替换后的文本和被替换的位置索引"""
        punctuation = string.punctuation
        if num_replacements is None:
            num_replacements = random.randint(1, len(text))
        
        indices = random.sample(range(1, len(text)-2), num_replacements)
        text_list = list(text)
        
        for i in indices:
            text_list[i] = random.choice(punctuation)
        
        return ''.join(text_list), indices


    def get_replacement_token_indices(self, original_text, modified_text, replaced_char_indices, tokenizer):
        """
        给定替换前后的文本、替换字符位置索引、tokenizer，
        返回替换字符在tokenized结果中对应的token索引。
        """
        # 编码时使用返回偏移信息
        encoded = tokenizer(modified_text, return_offsets_mapping=True, add_special_tokens=False)
        offset_mapping = encoded['offset_mapping']  # 每个token的起止字符索引
        replaced_token_indices = []

        # 遍历token offsets，看哪些token覆盖了被替换的字符索引
        for token_idx, (start, end) in enumerate(offset_mapping):
            for replaced_char_index in replaced_char_indices:
                if start <= replaced_char_index < end:
                    replaced_token_indices.append(token_idx)
                    break  # 同一个token最多记录一次
    
        return replaced_token_indices
    def intersection_ratio(self, tensorA, tensorB):
        # 转换为1维并去重
        A_set = torch.unique(tensorA.view(-1))
        B_set = torch.unique(tensorB.view(-1))

        # 找到交集
        common = torch.tensor([x.item() for x in B_set if x in A_set])

        # 计算比例
        ratio = common.numel() / B_set.numel()
        return 1-ratio
class Tokenized_data_chat(Dataset):
    def __init__(self, domains, tokenizer, total = -1, is_test = False) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.texts = []
        self.clip = total
        for domain in domains:
            domain_files = os.listdir(f'/home/jxzhou/PLM_PER/qwen/chat_txt_data/{domain}')
            d_file = sorted(domain_files)[0]
            
            f = open(f'/home/jxzhou/PLM_PER/qwen/chat_txt_data/{domain}/{d_file}')
            for _ in range(total):
                line = f.readline()
                self.texts.append(line)
            f.close()
        # random.shuffle(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # text = ''.join([self.texts[0:self.clip*1][index],self.texts[self.clip*1:self.clip*2][index]])
        
        text = self.texts[index]

        ids = self.tokenizer((text,), return_tensors="pt").input_ids
        # print(ids.shape)
        return ids
    



class Tokenized_data_locality(Dataset):
    def __init__(self, domains, tokenizer, total = -1, is_test = False) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.texts = []
        f = open(f'/home/jxzhou/PLM_PER/qwen/chat_txt_data/locality_data.txt')
        for _ in range(total):
            line = f.readline()
            self.texts.append(line)
        self.common = ['study',  'patterns' ,'systems', 'that', 'process', 'data', 'using', 'to', 'and', 'using']
        self.common = self.tokenizer(self.common, return_tensors="pt").input_ids
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # text = ''.join([self.texts[0:self.clip*1][index],self.texts[self.clip*1:self.clip*2][index]])
        
        text = self.texts[index]

        ids = self.tokenizer((text,), return_tensors="pt").input_ids
        # print(ids.shape)
        return ids




class OpenThoughtsDataset(Dataset):
    def __init__(self, dataset_dir, tokenizer, max_seq_len=2048, padding=True):
        super().__init__()
        self.data = load_from_disk(dataset_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        system = item.get("system", "")
        conversations = item["conversations"]

        # 拼接文本
        text = ""
        if system:
            text += f"System: {system.strip()}\n"
        for turn in conversations:
            role = turn["from"]
            content = turn["value"].strip()
            text += f"{role.capitalize()}: {content}\n"

        # 编码
        if self.padding:
            token_ids = self.tokenizer(
                text,
                padding='max_length',
                max_length=self.max_seq_len + 1,
                truncation=True,
                return_tensors='pt'
            ).input_ids
            real_len = len(self.tokenizer(text).input_ids)
        else:
            token_ids = self.tokenizer(text, return_tensors='pt').input_ids
            real_len = len(token_ids[0])

        return token_ids[0, :-1], token_ids[0, 1:], real_len
    



class DomainData_DCLM(Dataset):
    def __init__(self, dataset_dir, cuts, max_seq_len, tokenizer, padding=True) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        # 递归收集所有子文件夹中的jsonl文件
        self.files = []
        for root, _, filenames in os.walk(dataset_dir):
            for filename in filenames:
                if filename.endswith('.jsonl'):
                    self.files.append(os.path.join(root, filename))
        self.files = sorted(self.files)[cuts*2200:(cuts+1)*2200]
        # print("txt number",len(self.files))
        
        self._load_file(0)  # 加载第一个文件
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.combine_sequence = int(np.ceil(max_seq_len / 1024))
        self.lines_per_file = len(self.file_texts) // self.combine_sequence
        self.padding = padding

    def __len__(self):
        return len(self.files) * self.lines_per_file

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(self.files[self.cur_file_idx], 'r', encoding='utf-8') as f:
            # 直接读取每行作为文本，去除首尾空白字符
            self.file_texts = [line.strip() for line in f if line.strip()]
            # 过滤掉空行

    def __getitem__(self, index):
        file_idx = index // self.lines_per_file
        if file_idx != self.cur_file_idx:
            self._load_file(file_idx)
        
        # 计算行索引并确保不越界
        line_idx = index % self.lines_per_file
        max_line_idx = len(self.file_texts) // self.combine_sequence - 1
        line_idx = min(line_idx, max_line_idx)
        
        # 组合多个文本片段
        start_idx = line_idx * self.combine_sequence
        end_idx = start_idx + self.combine_sequence
        text = ''.join(self.file_texts[start_idx:end_idx])

        # 处理tokenization
        if self.padding:
            token_ids = self.tokenizer(
                text, 
                padding='max_length',
                max_length=self.max_seq_len + 1,
                padding_side='right',
                truncation=True,
                return_tensors='pt'
            ).input_ids
            real_len = len(self.tokenizer(text).input_ids)
        else:
            token_ids = self.tokenizer(text, return_tensors='pt').input_ids
            real_len = len(token_ids[0])
        
        return token_ids[0, :-1], token_ids[0, 1:], real_len