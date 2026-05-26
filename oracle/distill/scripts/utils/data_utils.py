import os
import numpy as np
import torch
import json

from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map
from datasets import load_from_disk
from datasets import Dataset as HFDataset
import random
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
            print(f'{index}, {line_idx}, {len(self.file_texts)}, {self.dataset_dir}/{self.files[self.cur_file_idx]}')
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


class DomainData(Dataset):
    def __init__(self, dataset_dir, domain_prefix, max_seq_len, tokenizer, padding = True) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.files = sorted([file for file in os.listdir(dataset_dir) if file.endswith('.txt')]) # 3414 * 6400 = 21,849,600 files train, test 60065 lines
        self.files = [file for file in self.files if file.startswith(domain_prefix)]
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
            print(f'{index}, {line_idx}, {len(self.file_texts)}, {self.dataset_dir}/{self.files[self.cur_file_idx]}')
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



class DomainData_20B(Dataset):
    def __init__(self, dataset_dir, domain_prefix, max_seq_len, tokenizer, padding = True) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.files = sorted([file for file in os.listdir(dataset_dir) if file.endswith('.txt')]) # 3414 * 6400 = 21,849,600 files train, test 60065 lines
        self.files = [file for file in self.files]
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
            print(f'{index}, {line_idx}, {len(self.file_texts)}, {self.dataset_dir}/{self.files[self.cur_file_idx]}')
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
        self.files = sorted(self.files)[cuts*1200:(cuts+1)*1200]
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




class DomainData_reviews(Dataset):
    def __init__(self, dataset_dir, domain_prefix, max_seq_len, tokenizer, padding = True) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.files = sorted([file for file in os.listdir(dataset_dir) if file.endswith('.txt')]) # 3414 * 6400 = 21,849,600 files train, test 60065 lines
        self.files = [file for file in self.files]
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
            print(f'{index}, {line_idx}, {len(self.file_texts)}, {self.dataset_dir}/{self.files[self.cur_file_idx]}')
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


import json
class SFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_dir, max_seq_len=1024):
        self.dataset_dir = dataset_dir
        self.files = sorted([dir for dir in os.listdir(dataset_dir) if dir.endswith('.jsonl')]) # 3414 * 6400 = 21,849,600 files train, test 60065 lines
        self._load_file(0)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.lines_per_file = 4096

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(f'{self.dataset_dir}/{self.files[self.cur_file_idx]}') as f:
            self.file_texts = f.readlines()

    def __len__(self):
        return (len(self.files) - 1) * self.lines_per_file

    def __getitem__(self, i):
        local_idx = i % self.lines_per_file
        if local_idx >= len(self.file_texts):
            print(f'{self.cur_file_idx}, {local_idx}, {len(self.file_texts)}, {self.dataset_dir}/{self.files[self.cur_file_idx]}')
            local_idx = len(self.file_texts)
        data = json.loads(self.file_texts[local_idx])
        
        prompt_only = f"Instruction: \n{data['instruction']} Answer: "
        prompt = prompt_only + data["answer"]

        prompt_len = self.tokenizer(prompt_only, truncation=True, max_length=self.max_seq_len, return_tensors="pt").input_ids[0]
        qa_len = self.tokenizer(prompt, truncation=True, max_length=self.max_seq_len, return_tensors="pt").input_ids[0]
        # print(f"p1 {prompt_len.shape}  {len(prompt_len)} p2 {qa_len.shape} {len(qa_len)}")
        prompt_len = len(prompt_len)
        qa_len=  len(qa_len)
        
        input_ids = self.tokenizer(prompt, padding='max_length', padding_side="right", truncation=True, max_length=self.max_seq_len, return_tensors="pt").input_ids[0]
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask 掉 instruction
        # labels[qa_len:] = -100      # mask 掉 padding

        return input_ids[:-1], labels[1:], prompt_len, qa_len

    
class OpenThoughtsSFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_dir, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 加载数据
        self.data = load_from_disk(dataset_dir)####'/inspire/hdd/project/yunweiyuhuifu/p-shangli/data/sft/Open-Thoughts-114k/default/train'

        # 拆分为单轮问答样本
        self.qa_pairs = []
        for item in self.data:
            system_prompt = item.get("system", "").strip()
            conversations = item["conversations"]
            for i in range(len(conversations) - 1):
                if conversations[i]["from"] == "user" and conversations[i + 1]["from"] == "assistant":
                    question = conversations[i]["value"].strip()
                    answer = conversations[i + 1]["value"].strip()
                    self.qa_pairs.append((system_prompt, question, answer))

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, i):
        system, question, answer = self.qa_pairs[i]

        # 构造输入字符串
        prompt_only = f"Instruction:\n{question}\nAnswer:"
        full_prompt = f"{prompt_only} {answer}"

        # 添加 system prompt（如果非空）
        if system:
            prompt_only = f"System:\n{system}\n" + prompt_only
            full_prompt = f"System:\n{system}\n" + full_prompt

        # 分别编码
        prompt_ids = self.tokenizer(prompt_only, truncation=True, max_length=self.max_seq_len, return_tensors="pt").input_ids[0]
        full_ids = self.tokenizer(full_prompt, truncation=True, max_length=self.max_seq_len, return_tensors="pt").input_ids[0]

        prompt_len = len(prompt_ids)
        qa_len = len(full_ids)

        input_ids = self.tokenizer(full_prompt, padding='max_length', truncation=True,
                                   max_length=self.max_seq_len + 1, return_tensors="pt").input_ids[0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask 掉 instruction
        # labels[qa_len:] = -100      # mask 掉 padding

        return input_ids[:-1], labels[1:], prompt_len, qa_len


class MMLUSFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_dir, max_seq_len=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = True

        # 加载 .arrow 数据
        self.data = load_from_disk(dataset_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]

        # breakpoint()
    
        # 构造输入格式：question + 选项
        prompt_only = question.strip()
        for i, option in enumerate(choices):
            label = chr(ord("A") + i)
            prompt_only += f"\n{label}. {option.strip()}"
        prompt_only += F'\nAnswer:'
        # 格式化答案：提取字母部分（如 'C'）
        # correct_letter = answer.strip().split()[-1]  # '2 C' → 'C'
        if isinstance(answer, int):
            correct_letter = chr(ord("A") + answer)  # 0 → A, 1 → B, ...
        elif isinstance(answer, str):
            correct_letter = answer.strip().split()[-1]
        else:
            raise ValueError(f"Unsupported answer format: {answer}")

        full_prompt = prompt_only + f" {correct_letter}"

        prompt_ids = self.tokenizer(prompt_only, truncation=True, max_length=self.max_seq_len, return_tensors="pt").input_ids[0]
        full_ids = self.tokenizer(full_prompt, truncation=True, max_length=self.max_seq_len, return_tensors="pt").input_ids[0]

        prompt_len = len(prompt_ids)
        qa_len = len(full_ids)

        input_ids = self.tokenizer(full_prompt, padding='max_length', truncation=True,
                                   max_length=self.max_seq_len + 1, return_tensors="pt").input_ids[0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask 掉 instruction
        # labels[qa_len:] = -100      # mask 掉 padding

        return input_ids[:-1], labels[1:], prompt_len, qa_len

class Benchmark_ultrachat_FewShot_Qwen3(Dataset):
    def __init__(self, dataset_path, tokenizer, max_seq_len=4096, padding=True, n_shots=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.n_shots = n_shots

        # 加载 .arrow 数据
        self.data = HFDataset.load_from_disk('/inspire/hdd/project/yunweiyuhuifu/p-shangli/data/ultrachat/ultrachat/test_gen')

    def __len__(self):
        return len(self.data)

    def extract_last_assistant_answer(self, messages):
        # 从消息中提取最后一个 assistant 回复
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "").strip()
        return ""

    def format_example(self, prompt, answer=None, include_answer=True):
        text = f"Instruction: {prompt.strip()}\nAnswer:"
        if include_answer and answer is not None:
            text += f"\n{answer.strip()}"
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        messages = item["messages"]
        answer = self.extract_last_assistant_answer(messages)

        # Few-shot 示例（排除当前样本）
        shot_indices = list(range(len(self.data)))
        shot_indices.remove(idx)
        shots = random.sample(shot_indices, min(self.n_shots, len(shot_indices)))

        few_shot_prompt = ""
        for i in shots:
            ex = self.data[i]
            shot_prompt = ex["prompt"]
            shot_answer = self.extract_last_assistant_answer(ex["messages"])
            few_shot_prompt += self.format_example(shot_prompt, shot_answer, include_answer=True) + "\n\n"

        # 当前示例（不带答案）
        query_prompt = self.format_example(prompt, include_answer=False)

        # 拼接最终 prompt
        full_prompt = few_shot_prompt + "\n" + query_prompt

        # 普通编码方式
        input_enc = self.tokenizer(
            full_prompt,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        # Chat 模型格式（Qwen 专用）
        messages = [{"role": "user", "content": full_prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([chat_text], return_tensors="pt")

        return {
            "model_inputs": model_inputs,
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": answer,
            "question": prompt,
            "sentence": full_prompt + f"\nAnswer: {answer}",  # for debug
        }