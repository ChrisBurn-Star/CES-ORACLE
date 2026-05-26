from torch.utils.data import DataLoader
import torchvision
import multiprocessing
from transformers import BertConfig, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets, load_from_disk
import os
from collections import Counter
from itertools import chain
import torch
import numpy as np
import sys
import json
import pandas as pd

class Cifar10():
    def __init__(self, config=None) -> None:
        transform_argumented = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.batch_size = 64
        self.num_classes = 10

        train_dataset = torchvision.datasets.CIFAR10('datasets/cifar10', train=True, download=True, transform=transform_argumented)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = torchvision.datasets.CIFAR10('datasets/cifar10', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

class Cifar100():
    def __init__(self, config=None) -> None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.batch_size = 64
        self.num_classes = 100

        train_dataset = torchvision.datasets.CIFAR100('datasets/cifar100', train=True, download=True, transform=train_transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = torchvision.datasets.CIFAR100('datasets/cifar100', train=False, download=True, transform=test_transform)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

class Mnist():
    def __init__(self, config=None) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.batch_size = 256
        self.num_classes = 10

        train_dataset = torchvision.datasets.MNIST('datasets/mnist', train=True, download=True, transform=transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = torchvision.datasets.MNIST('datasets/mnist', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

class Wikitext():
    def group_texts(self, examples):
        block_size = self.block_size

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def preprocess(self, config, path):
        num_proc = multiprocessing.cpu_count() // 2

        raw_datasets = load_dataset('/home/jxzhou/PLM_PER/wikitext', config.dataset_name)
        tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text']), batched=True, num_proc=num_proc, remove_columns=["text"])
        lm_dataset = tokenized_datasets.map(self.group_texts, batched=True)
        lm_dataset.save_to_disk(path)
        return lm_dataset

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path = os.path.join('/home/jxzhou/datasets/wikitxt2forLM', str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
class Wikitext_103():
    def group_texts(self, examples):
        block_size = self.block_size

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        num_proc = multiprocessing.cpu_count() // 2

        raw_datasets = load_dataset('/home/jxzhou/PLM_PER/wikitxt-103')

        # tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text']), batched=True, num_proc=num_proc, remove_columns=["text"])
        # lm_dataset = tokenized_datasets.map(self.group_texts, batched=True)



        column_names = raw_datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = raw_datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        # lm_dataset.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path = os.path.join('/home/jxzhou/datasets/wikitxt103forLM', str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        print(len(lm_datasets['train']))
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class IMDB():
    def group_texts(self, examples):
        block_size = self.block_size
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys() if k != 'label'}
        total_length = len(concatenated_examples[list(examples.keys())[1]])
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def __init__(self, config) -> None:
        self.block_size = config.seq_len
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        self.batch_size = config.batch_size

        raw_datasets = load_dataset('imdb')
        tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text'], padding='max_length', truncation=True), batched=True, num_proc=16, remove_columns=["text"])

        path = os.path.join(config.dataset_cache[config.dataset_name], str(self.block_size))
        if not config.preprocessed:
            lm_datasets = tokenized_datasets.map(self.group_texts, batched=True, remove_columns=['label'])
            lm_datasets.save_to_disk(path)
        lm_datasets = load_from_disk(path)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.lm_train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.lm_val_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)
        pass

class AGNews():
    def group_texts(self, examples):
        block_size = self.block_size
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys() if k != 'label'}
        total_length = len(concatenated_examples[list(examples.keys())[1]])
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def lt_dataset(self, tokenized_datasets, tokenizer, ratio=.3):
        all_ids = [sample['input_ids'] for sample in tokenized_datasets['train']]
        concat_ids = list(chain(*all_ids))
        freqs = Counter(concat_ids)

        train_freq = []
        for sample in tokenized_datasets['train']:
            freq = [freqs[w] for w in sample['input_ids'] if w not in tokenizer.all_special_ids]
            train_freq.append(sum(freq) / len(freq))
        _, tail_indices = torch.topk(torch.tensor(train_freq), k=int(ratio*len(train_freq)), largest=False)
        lt_train = Dataset.from_dict(tokenized_datasets['train'][tail_indices])
        lt_train.set_format("torch")
        
        test_freq = []
        for sample in tokenized_datasets['test']:
            freq = [freqs[w] for w in sample['input_ids'] if w not in tokenizer.all_special_ids]
            test_freq.append(sum(freq) / len(freq))
        _, tail_indices = torch.topk(torch.tensor(test_freq), k=int(ratio*len(test_freq)), largest=False)
        lt_test = Dataset.from_dict(tokenized_datasets['test'][tail_indices])
        lt_test.set_format("torch")

        return lt_train, lt_test

    def __init__(self, config) -> None:
        self.block_size = config.seq_len
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        self.batch_size = config.batch_size

        raw_datasets = load_dataset('/home/jxzhou/PLM_PER/ag_news', split=['train[:20%]', 'test[:50%]', 'test[:80%]'])
        raw_datasets = DatasetDict({name: dataset for name, dataset in zip(['train', 'test', 'val'], raw_datasets)})
        tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text'], padding='max_length', truncation=True), batched=True, num_proc=16, remove_columns=["text"])

        path = os.path.join("/home/jxzhou/datasets/AGNewsforLM", str(self.block_size))
        if not config.preprocessed:
            lm_datasets = tokenized_datasets.map(self.group_texts, batched=True, remove_columns=['label'])
            lm_datasets.save_to_disk(path)
        lm_datasets = load_from_disk(path)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['val'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

        tokenized_datasets.set_format("torch")
        self.tor_train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        self.tor_test_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)

        lt_train, lt_test = self.lt_dataset(tokenized_datasets, self.tokenizer)
        self.lt_train_loader = DataLoader(lt_train, batch_size=self.batch_size, shuffle=True)
        self.lt_test_loader = DataLoader(lt_test, batch_size=self.batch_size, shuffle=False)
        pass



class Shape3D():
    def __init__(self):
        '''
        tag in {base, vertex, edge, cube}
        '''
        image_path = 'datasets/3DShape/all_image.npy'
        label_path = 'datasets/3DShape/all_label.npy'
        images = np.load(image_path)
        label = np.load(label_path)
        
        self.data = torch.from_numpy(images).permute([0,3,1,2]).float() / 255.0  # N, C, W, H
        self.label = torch.from_numpy(label)
        self.label[:, 0] = self.label[:, 0] / 0.6
        self.label[:, 1] = self.label[:, 1] / 0.6
        self.label[:, 2] = self.label[:, 2] / 0.6
        self.label[:, 3] = (self.label[:, 3]  - 0.75) / 0.25
        self.label[:, 4] = self.label[:, 4] / 2
        self.label[:, 5] = (self.label[:, 5] + 15) / 30
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class RestaurantForLM():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/jxzhou/PLM_PER/PRE-TRAIN-DATA/post_train/yelp_restaurant.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{1}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class ACLForLM():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/jxzhou/PLM_PER/PRE-TRAIN-DATA/post_train/acl_anthology.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{1}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path = os.path.join("/home/jxzhou/datasets/ACLforLM64", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class PhoneForLM():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/jxzhou/PLM_PER/PRE-TRAIN-DATA/post_train/phone.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{5}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class CameraForLM():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/jxzhou/PLM_PER/PRE-TRAIN-DATA/post_train/camera.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{5}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class ReviewtForLM():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/jxzhou/PLM_PER/data2/review.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{1}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class LegalForLM():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/jxzhou/PLM_PER/data2/legal.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{10}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{10}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        print(len(lm_datasets['train']),len(lm_datasets['validation']))

        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class PubMedForLM0():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/archen/format_pubmed_small.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[{5}%:]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        path = os.path.join("/home/archen/datasets/PubMedforLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class AIForLM():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/archen/ai_corpus.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{5}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        path = os.path.join("/home/archen/datasets/AIforLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class MixedData():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        
        lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        lm_datasets3 = load_from_disk(path3)
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(2000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(2000))
        lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(2000))
        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(2000))
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(2000))

        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(100))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(100))
        lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(100))
        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(100))
        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(100))

        print(len(lm_datasets1t), len(lm_datasets2t), len(lm_datasets3t), len(lm_datasets4t), len(lm_datasets5t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1t, lm_datasets2t, lm_datasets3t,lm_datasets4t,lm_datasets5t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1v, lm_datasets2v, lm_datasets3v,lm_datasets4v,lm_datasets5v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class MixedData_stage1():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        
        lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        lm_datasets3 = load_from_disk(path3)
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(2000,12000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(2000,12000))
        lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(2000,12000))
        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(2000,12000))
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(2000,12000))

        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(100))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(100))
        lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(100))
        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(100))
        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(100))

        print(len(lm_datasets1t), len(lm_datasets2t), len(lm_datasets3t), len(lm_datasets4t), len(lm_datasets5t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1t, lm_datasets2t, lm_datasets3t,lm_datasets4t,lm_datasets5t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1v, lm_datasets2v, lm_datasets3v,lm_datasets4v,lm_datasets5v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class old_MixedData_after_stage1():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        
        lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        lm_datasets3 = load_from_disk(path3)
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(12000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(12000))
        lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(12000))
        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(12000))
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(12000))

        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(100))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(100))
        lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(100))
        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(100))
        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(100))

        print(len(lm_datasets1t), len(lm_datasets2t), len(lm_datasets3t), len(lm_datasets4t), len(lm_datasets5t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1t, lm_datasets2t, lm_datasets3t,lm_datasets4t,lm_datasets5t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1v, lm_datasets2v, lm_datasets3v,lm_datasets4v,lm_datasets5v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class Mixdata_1103():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        
        lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        lm_datasets3 = load_from_disk(path3)
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(10000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(10000))
        lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(10000))
        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(10000))
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(10000))

        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(500))
        lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(500))
        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(500))
        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))

        print(len(lm_datasets1t), len(lm_datasets2t), len(lm_datasets3t), len(lm_datasets4t), len(lm_datasets5t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1t, lm_datasets2t, lm_datasets3t,lm_datasets4t,lm_datasets5t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1v, lm_datasets2v, lm_datasets3v,lm_datasets4v,lm_datasets5v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class Mixdata_1213():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        
        lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        lm_datasets3 = load_from_disk(path3)
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(10000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(10000))
        lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(10000))
        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(10000))
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(10000))

        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(500))
        lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(500))
        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(500))
        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))

        print(len(lm_datasets1t), len(lm_datasets2t), len(lm_datasets3t), len(lm_datasets4t), len(lm_datasets5t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1t, lm_datasets2t, lm_datasets3t,lm_datasets4t,lm_datasets5t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1v, lm_datasets2v, lm_datasets3v,lm_datasets4v,lm_datasets5v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class Mixdata_1115():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        
        lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        lm_datasets3 = load_from_disk(path3)
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(5000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(5000))
        lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(5000))
        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(5000))
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(5000))

        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(500))
        lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(500))
        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(500))
        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))

        print(len(lm_datasets1t), len(lm_datasets2t), len(lm_datasets3t), len(lm_datasets4t), len(lm_datasets5t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1t, lm_datasets2t, lm_datasets3t,lm_datasets4t,lm_datasets5t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1v, lm_datasets2v, lm_datasets3v,lm_datasets4v,lm_datasets5v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
class MixedData_1121():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))
        
        lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        lm_datasets3 = load_from_disk(path3)
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(10))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(10))
        lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(10))
        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(100))
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(100))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6['train'], range(10))

        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(500))
        lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(500))
        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(500))
        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(500))

        print(len(lm_datasets1t), len(lm_datasets2t), len(lm_datasets3t), len(lm_datasets4t), len(lm_datasets5t), len(lm_datasets6t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1t, lm_datasets2t, lm_datasets3t,lm_datasets4t,lm_datasets5t, lm_datasets6t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1v, lm_datasets2v, lm_datasets3v,lm_datasets4v,lm_datasets5v, lm_datasets6v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = Data


class MixedData_1211_0():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        # path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        # path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        # path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        # path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))
        
        # lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        # lm_datasets3 = load_from_disk(path3)
        # lm_datasets4 = load_from_disk(path4)
        # lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)
        # lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(2000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(1000))
        # lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(2000))
        # lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(2000))
        # lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(2000))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6['train'], range(1000))

        # lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(500))
        # lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(500))
        # lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(500))
        # lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(500))

        # print(len(lm_datasets1t), len(lm_datasets2t), len(lm_datasets3t), len(lm_datasets4t), len(lm_datasets5t), len(lm_datasets6t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets2t,lm_datasets6t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets2v, lm_datasets6v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)


class MixedData_1211_1():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        # path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        # path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        # path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        # path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))
        
        # lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        # lm_datasets3 = load_from_disk(path3)
        # lm_datasets4 = load_from_disk(path4)
        # lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)
        # lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(2000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(1000,10000))
        # lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(2000))
        # lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(2000))
        # lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(2000))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6['train'], range(1000,10000))

        # lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(500))
        # lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(500))
        # lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(500))
        # lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(500))

        # print(len(lm_datasets1t), len(lm_datasets2t), len(lm_datasets3t), len(lm_datasets4t), len(lm_datasets5t), len(lm_datasets6t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets2t,lm_datasets6t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets2v, lm_datasets6v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)


class MixedData_0110_0():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        # path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        # path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        # path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        # path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))
        
        # lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        # lm_datasets3 = load_from_disk(path3)
        # lm_datasets4 = load_from_disk(path4)
        # lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)
        # lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(2000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(2500))
        # lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(2000))
        # lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(2000))
        # lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(2000))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6['train'], range(2500))

        # lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(500))
        # lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(500))
        # lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(500))
        # lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(500))

        # print(len(lm_datasets1t), len(lm_datasets2t), len(lm_datasets3t), len(lm_datasets4t), len(lm_datasets5t), len(lm_datasets6t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets2t,lm_datasets6t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets2v, lm_datasets6v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)


class MixedData_0110_1():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        # path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        # path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        # path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        # path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))
        
        # lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        # lm_datasets3 = load_from_disk(path3)
        # lm_datasets4 = load_from_disk(path4)
        # lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)
        # lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(2000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(2500,10000))
        # lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(2000))
        # lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(2000))
        # lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(2000))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6['train'], range(2500,10000))

        # lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(500))
        # lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(500))
        # lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(500))
        # lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(500))

        print(len(lm_datasets2t), len(lm_datasets6t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets2t,lm_datasets6t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets2v, lm_datasets6v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)




class MoMoE_MIXED_WIKI103_0124_3():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path4 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))
        

        lm_datasets4 = load_from_disk(path4)


        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(0,800000,2000))
    
        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(32))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets4t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets4v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)

class MoMoE_MIXED_WIKI103_0124_2():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path4 = os.path.join("/home/jxzhou/datasets/PubMedForLM", str(self.block_size))
        

        lm_datasets4 = load_from_disk(path4)


        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(128))
    
        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(32))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets4t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets4v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)

class MoMoE_MIXED_WIKI103_0124_1():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path4 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        

        lm_datasets4 = load_from_disk(path4)


        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(100256,100512))
    
        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(32))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets4t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets4v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)



class MoMoE_MIXED_WIKI103_0124():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        # path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        # path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        # path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))
        
        # lm_datasets1 = load_from_disk(path1)
        # lm_datasets2 = load_from_disk(path2)
        # lm_datasets3 = load_from_disk(path3)
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)
        # lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(2000))
        # lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(2000))
        # lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(2000))
        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(400000))
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(400000))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6["train"], range(10000))

        # lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(100))
        # lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(100))
        # lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(100))
        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(500))
        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(500))

        print(len(lm_datasets6t),len(lm_datasets5t),len(lm_datasets4t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets6t,lm_datasets4t,lm_datasets5t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets6v,lm_datasets5v,lm_datasets4v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
class MoMoE_WIKI103_WARMUP():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))


        
        lm_datasets1 = load_from_disk(path1)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(10000))


        print(len(lm_datasets1t),len(lm_datasets1["validation"]))
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets1t, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets1["validation"], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
class MoMoE_WIKI103_TOSHOW():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))


        
        lm_datasets1 = load_from_disk(path1)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(128))


        print(len(lm_datasets1t),len(lm_datasets1["validation"]))
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets1t, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets1["validation"], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)

class MoMoE_MIXED():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        # path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))

        
        lm_datasets1 = load_from_disk(path2)
        lm_datasets2 = load_from_disk(path6)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(400000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(400000))
        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['train'], range(600))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['train'], range(600))
        print(len(lm_datasets1t),len(lm_datasets1v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1t,lm_datasets2t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1v, lm_datasets2v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)

class MoMoE_MIXED_RES():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        # path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))

        
        lm_datasets1 = load_from_disk(path2)
        lm_datasets2 = load_from_disk(path6)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(400000))
        # lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(400000))
        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['train'], range(600))
        # lm_datasets2v = torch.utils.data.Subset(lm_datasets2['train'], range(600))
        print(len(lm_datasets1t),len(lm_datasets1v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)



class MoMoE_MIXED_ACL():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        # path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))

        
        lm_datasets1 = load_from_disk(path2)
        lm_datasets2 = load_from_disk(path6)
        # lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(400000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(400000))
        # lm_datasets1v = torch.utils.data.Subset(lm_datasets1['train'], range(600))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['train'], range(600))
        print(len(lm_datasets2t),len(lm_datasets2v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets2t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets2v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)

class MoMoE_MIXED_REV():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        # path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))

        
        lm_datasets1 = load_from_disk(path2)
        lm_datasets2 = load_from_disk(path6)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(400000))
        # lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(400000))
        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['train'], range(600))
        # lm_datasets2v = torch.utils.data.Subset(lm_datasets2['train'], range(600))
        print(len(lm_datasets1t),len(lm_datasets1v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)



class MoMoE_MIXED_LEG():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        # path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))

        
        lm_datasets1 = load_from_disk(path2)
        lm_datasets2 = load_from_disk(path6)
        # lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(400000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(400000))
        # lm_datasets1v = torch.utils.data.Subset(lm_datasets1['train'], range(600))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['train'], range(600))
        print(len(lm_datasets2t),len(lm_datasets2v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets2t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets2v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)

class MoMoE_MIXED_0128():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        # path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))

        
        lm_datasets1 = load_from_disk(path2)
        lm_datasets2 = load_from_disk(path6)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(400000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(400000))
        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['train'], range(600))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['train'], range(600))
        print(len(lm_datasets1t),len(lm_datasets2t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets2t,lm_datasets1t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets2v,lm_datasets2v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)




class MoMoE_MIXED_WARMUP():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        # path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))
        # path2 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        # path6 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        
        lm_datasets1 = load_from_disk(path2)
        lm_datasets2 = load_from_disk(path6)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(2000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(2000))
        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['train'], range(500))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['train'], range(500))
        print(len(lm_datasets1t),len(lm_datasets1v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1t,lm_datasets2t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1v, lm_datasets2v])
        
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)

class MoMoE_MIXED_TOSHOW():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        # path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))

        
        lm_datasets1 = load_from_disk(path2)
        lm_datasets2 = load_from_disk(path6)
        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(64))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(64))
        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['train'], range(500))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['train'], range(500))
        print(len(lm_datasets1t),len(lm_datasets1v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1t,lm_datasets2t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1v, lm_datasets2v])
        
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)


class MoMoE_WIKI103():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))

        
        lm_datasets1 = load_from_disk(path1)


        print(len(lm_datasets1["train"]),len(lm_datasets1["validation"]))
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets1["train"], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets1["validation"], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)


class MixedData_1211():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        # path1 = os.path.join("/home/jxzhou/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        # path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        # path4 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        # path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))
        
        # lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        # lm_datasets3 = load_from_disk(path3)
        # lm_datasets4 = load_from_disk(path4)
        # lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)
        # lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(2000))
        lm_datasets2t = torch.utils.data.Subset(lm_datasets2['train'], range(10000))
        # lm_datasets3t = torch.utils.data.Subset(lm_datasets3['train'], range(2000))
        # lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(2000))
        # lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(2000))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6['train'], range(10000))

        # lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))
        lm_datasets2v = torch.utils.data.Subset(lm_datasets2['validation'], range(500))
        # lm_datasets3v = torch.utils.data.Subset(lm_datasets3['validation'], range(500))
        # lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(500))
        # lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(500))

        print(len(lm_datasets2t), len(lm_datasets6t))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets2t,lm_datasets6t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets2v, lm_datasets6v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)

class RestaurantforLM_1103():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(10000))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
class RestaurantforLM_0109():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(32))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)



class Review_1103():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(10000))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class ACLForLM_1103():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(10000))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
class ACLForLM_0109():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(32))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class Wikitxt2ForLM_1103():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/wikitxt2forLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(10000))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
class Wikitxt103ForLM_1103():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(10000))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class Wikitxt103ForLM_0109():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(32))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class Wikitxt103ForLM_0102_warmup():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(5000))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class Wikitxt103ForLM_0102_rose():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(5000,20000))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class Wikitxt103ForLM_0102_bert():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(20000))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class Wikitxt103ForLM_80W():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(800000))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['validation'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class AGNewsForLM_1103():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        path1 = os.path.join("/home/jxzhou/datasets/AGNewsforLM", str(self.block_size))

        lm_datasets1 = load_from_disk(path1)

        lm_datasets1t = torch.utils.data.Subset(lm_datasets1['train'], range(10000))


        lm_datasets1v = torch.utils.data.Subset(lm_datasets1['val'], range(500))


        print(len(lm_datasets1t))
        lm_datasets = lm_datasets1t
        lm_datasets_val = lm_datasets1v
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class Eval_Data1():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        path1 = os.path.join("/home/archen/datasets/restaurantforLM", str(self.block_size))
        path2 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))
        lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        
        lm_datasets1 = torch.utils.data.Subset(lm_datasets1['train'], range(100))
        lm_datasets2 = torch.utils.data.Subset(lm_datasets2['train'], range(100))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets1, lm_datasets2])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader1 = DataLoader(lm_datasets1, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.train_loader2 = DataLoader(lm_datasets2, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.mixed_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class Eval_Data2():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))

        path1 = os.path.join("/home/archen/datasets/PhoneforLM", str(self.block_size))
        path2 = os.path.join("/home/archen/datasets/restaurantforLM", str(self.block_size))
        path3 = os.path.join("/home/archen/datasets/CameraforLM", str(self.block_size))
        lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        lm_datasets3 = load_from_disk(path3)
        print(len(lm_datasets1['train']), len(lm_datasets2['train']), len(lm_datasets3['train']))

        # path2 = os.path.join("/home/archen/datasets/restaurantforLM", str(self.block_size))
        # lm_datasets1 = load_from_disk(path1)
        # lm_datasets2 = load_from_disk(path2)
        lm_datasets1 = torch.utils.data.Subset(lm_datasets1['train'], range(3000))
        lm_datasets2 = torch.utils.data.Subset(lm_datasets2['train'], range(3000))
        lm_datasets3 = torch.utils.data.Subset(lm_datasets3['train'], range(3000))
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader1 = DataLoader(lm_datasets1, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.train_loader2 = DataLoader(lm_datasets2, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.train_loader3 = DataLoader(lm_datasets3, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class RestaurantForLM_small():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/jxzhou/PLM_PER/PRE-TRAIN-DATA/post_train/yelp_restaurant.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{1}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            #num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            #load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            #num_proc=config.preprocessing_num_workers,
            #load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path = os.path.join("/home/jxzhou/PLM_PER/PRE-TRAIN-DATA/restaurantforLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        lm_datasets_train = torch.utils.data.Subset(lm_datasets['train'], range(19200))
        lm_datasets_val = torch.utils.data.Subset(lm_datasets['validation'], range(1920))
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets_train, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class ACLForLM_small():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/jxzhou/PLM_PER/PRE-TRAIN-DATA/post_train/acl_anthology.txt'}
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[:{1}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[{5}%:]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            #num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            #load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            #num_proc=config.preprocessing_num_workers,
            #load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path = os.path.join("/home/jxzhou/PLM_PER/PRE-TRAIN-DATA/ACLforLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        lm_datasets_train = torch.utils.data.Subset(lm_datasets['train'], range(19200))
        lm_datasets_val = torch.utils.data.Subset(lm_datasets['validation'], range(1920))
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets_train, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class CustomDataset(Dataset):
    def __init__(self, data):
        self._data = data
    
    def __len__(self):
        return len(self._data['input_ids'])
    
    def __getitem__(self, idx):
        sample = {'input_ids': self._data['input_ids'][idx], 'labels': self._data['labels'][idx], 'attention_mask': self._data['attention_mask'][idx]}
        return sample
    
class ReplayDataset():
    def __init__(self, batch_size, path):
        tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        replay_data = torch.load(path, map_location='cpu')        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        
        self.replay_loader = {}
        
        for key, input_ids in replay_data.items(): 
            # print(input_ids.dtype)
            input_ids = input_ids.to(torch.long)            
            special_tokens_mask = [tokenizer.get_special_tokens_mask(input_id, already_has_special_tokens=True) for input_id in input_ids]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            # print(special_tokens_mask.dtype)
            inputs, labels = data_collator.torch_mask_tokens(input_ids, special_tokens_mask)
            # print(labels.shape)
            attention_mask = torch.ones(inputs.shape[0], inputs.shape[1])
            data = {'input_ids': inputs, 'labels': labels, 'attention_mask': attention_mask}
            replay_dataset = CustomDataset(data)
            # print(len(replay_dataset))
            replay_loader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=True)            
            self.replay_loader[key] = replay_loader


class PubMedForLM():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/PUBMED/train/train2.txt','val': '/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/PUBMED/val/val2.txt'}

        
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"val[:{100}%]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[:{100}%]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path = os.path.join("/home/jxzhou/datasets/PubMedForLM", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        print(len(lm_datasets['train']),len(lm_datasets['validation']))
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        print(len(lm_datasets['train']),len(lm_datasets['validation']))
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class DIAGNOSIS():
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        data_files = {'train': '/home/jxzhou/PLM_PER/MoMoE_rawdata/MIMIC-III/DIAGNOSIS/anonymized_patient_notes.txt'}

        
        datasets = load_dataset('text', data_files=data_files)
        datasets["validation"] = load_dataset(
            'text', data_files=data_files,split=f"train[{85}%:]"
        )
        datasets["train"] = load_dataset(
            'text', data_files=data_files,
            split=f"train[:{75}%]",
        )
        #rawdatasets
        column_names = datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path = os.path.join("/home/jxzhou/datasets/DIAGNOSIS", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        print(len(lm_datasets['train']),len(lm_datasets['validation']))
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)




class Test_Data():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        # path1 = os.path.join("/home/archen/datasets/ACLforLM", str(self.block_size))

        path1 = os.path.join("/home/jxzhou/datasets/restaurantforLM", str(self.block_size))
        path2 = os.path.join("/home/jxzhou/datasets/ACLforLM", str(self.block_size))
        path3 = os.path.join("/home/jxzhou/datasets/CameraforLM", str(self.block_size))
        lm_datasets1 = load_from_disk(path1)
        lm_datasets2 = load_from_disk(path2)
        lm_datasets3 = load_from_disk(path3)

        
        lm_datasets1_warmup = torch.utils.data.Subset(lm_datasets1['train'], range(3000))
        lm_datasets2_warmup = torch.utils.data.Subset(lm_datasets2['train'], range(3000))
        lm_datasets1_warmup_val = torch.utils.data.Subset(lm_datasets1['validation'], range(500))
        lm_datasets2_warmup_val = torch.utils.data.Subset(lm_datasets2['validation'], range(500))
        lm_datasets3_warmup = torch.utils.data.Subset(lm_datasets3['train'], range(3000))
        lm_datasets3_warmup_val = torch.utils.data.Subset(lm_datasets3['validation'], range(500))
        lm_datasets1_train = torch.utils.data.Subset(lm_datasets1['train'], range(3000,13000))
        lm_datasets2_train = torch.utils.data.Subset(lm_datasets2['train'], range(3000,13000))
        lm_datasets3_train = torch.utils.data.Subset(lm_datasets3['train'], range(3000,13000))
        # print(len(lm_datasets1['validation']), len(lm_datasets2['validation']))
        lm_datasets1_val = torch.utils.data.Subset(lm_datasets1['validation'], range(500, 2000))
        lm_datasets2_val = torch.utils.data.Subset(lm_datasets2['validation'], range(500, 2000))
        lm_datasets3_val = torch.utils.data.Subset(lm_datasets3['validation'], range(500, 2000))


        lm_datasets_warmup = torch.utils.data.ConcatDataset([lm_datasets1_warmup, lm_datasets2_warmup])
        lm_datasets_train = torch.utils.data.ConcatDataset([lm_datasets1_train, lm_datasets2_train])
        lm_datasets_warmup_val = torch.utils.data.ConcatDataset([lm_datasets1_warmup_val, lm_datasets2_warmup_val])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets1_val, lm_datasets2_val])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader_warmup = DataLoader(lm_datasets_warmup, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader_warmup = DataLoader(lm_datasets_warmup_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader = DataLoader(lm_datasets_train, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader2 = DataLoader(lm_datasets3_train, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)

        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader2 = DataLoader(lm_datasets3_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.val_loader1 = DataLoader(lm_datasets1_val, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader2 = DataLoader(lm_datasets2_val, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class MoMoE_MIXED_LEGAL_PUBMED():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/PubMedForLM", str(self.block_size))
        

        lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)
        
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(1000000))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6["train"], range(1000000))


        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(1000))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(1000))

        # lm_datasets5v_f = torch.utils.data.Subset(lm_datasets5['validation'], range(500))
        # lm_datasets6v_f = torch.utils.data.Subset(lm_datasets6['validation'], range(500))

        print(len(lm_datasets6t),len(lm_datasets6v),len(lm_datasets5t),len(lm_datasets5v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets6t,lm_datasets5t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets6v,lm_datasets5v])
        # lm_datasets_val_f = torch.utils.data.ConcatDataset([lm_datasets5v_f,lm_datasets6v_f])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.val_loader_f = DataLoader(lm_datasets_val_f, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)



class MoMoE_longtailed():
    def __init__(self, config,KKK):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        
        
        path4 = os.path.join("/home/jxzhou/datasets/wikipediaforLM", str(self.block_size))
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/PubMedForLM", str(self.block_size))
        
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)

        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(500000))
        # lm_datasets4t = lm_datasets4['train']

        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(KKK))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6["train"], range(KKK))

        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(5000))
        # lm_datasets4v = lm_datasets4['validation']

        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(1000))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(1000))

        lm_datasets5v_ntk = torch.utils.data.Subset(lm_datasets5['validation'], range(32))
        lm_datasets6v_ntk = torch.utils.data.Subset(lm_datasets6['validation'], range(32))


        print(len(lm_datasets6t),len(lm_datasets6v),len(lm_datasets5t),len(lm_datasets5v),len(lm_datasets4t),len(lm_datasets4v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets5t,lm_datasets4t,lm_datasets6t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets6v,lm_datasets5v,lm_datasets4v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader1 = DataLoader(lm_datasets5t, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader2 = DataLoader(lm_datasets6t, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader3 = DataLoader(lm_datasets4t, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader_domain1 = DataLoader(lm_datasets5v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader_domain2 = DataLoader(lm_datasets6v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader_domain_common = DataLoader(lm_datasets4v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader_domain1_ntk = DataLoader(lm_datasets5v_ntk, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader_domain2_ntk = DataLoader(lm_datasets6v_ntk, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class MoMoE_rebuttal():
    def __init__(self, config,KKK):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        
        
        path4 = os.path.join("/home/jxzhou/datasets/wikipediaforLM", str(self.block_size))
        # path6 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        
        lm_datasets4 = load_from_disk(path4)
        # lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)

        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(5000))
        # lm_datasets4t = lm_datasets4['train']

        # lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(KKK))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6["train"], range(KKK))

        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(5000))
        # lm_datasets4v = lm_datasets4['validation']

        # lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(1000))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(1000))

        # lm_datasets5v_ntk = torch.utils.data.Subset(lm_datasets5['validation'], range(32))
        lm_datasets6v_ntk = torch.utils.data.Subset(lm_datasets6['validation'], range(32))


        print(len(lm_datasets6t),len(lm_datasets6v),len(lm_datasets4t),len(lm_datasets4v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets4t,lm_datasets6t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets6v,lm_datasets4v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader1 = DataLoader(lm_datasets5t, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader2 = DataLoader(lm_datasets6t, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader3 = DataLoader(lm_datasets4t, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.val_loader_domain1 = DataLoader(lm_datasets5v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader_domain2 = DataLoader(lm_datasets6v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader_domain_common = DataLoader(lm_datasets4v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.val_loader_domain1_ntk = DataLoader(lm_datasets5v_ntk, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader_domain2_ntk = DataLoader(lm_datasets6v_ntk, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)



class MoE_huawei():
    def __init__(self, config,KKK):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        
        
        path4 = os.path.join("/home/jxzhou/datasets/wikipediaforLM", str(self.block_size))
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/PubMedForLM", str(self.block_size))
        
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)

        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(1000000))
        # lm_datasets4t = lm_datasets4['train']

        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(KKK))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6["train"], range(KKK))

        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(1000))
        # lm_datasets4v = lm_datasets4['validation']

        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(1000))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(1000))

        lm_datasets5v_ntk = torch.utils.data.Subset(lm_datasets5['validation'], range(32))
        lm_datasets6v_ntk = torch.utils.data.Subset(lm_datasets6['validation'], range(32))


        print(len(lm_datasets6t),len(lm_datasets6v),len(lm_datasets5t),len(lm_datasets5v),len(lm_datasets4t),len(lm_datasets4v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets5t,lm_datasets4t,lm_datasets6t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets6v,lm_datasets5v,lm_datasets4v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader1 = DataLoader(lm_datasets5t, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader2 = DataLoader(lm_datasets6t, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader3 = DataLoader(lm_datasets4t, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader_domain1 = DataLoader(lm_datasets5v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader_domain2 = DataLoader(lm_datasets6v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader_domain_common = DataLoader(lm_datasets4v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader_domain1_ntk = DataLoader(lm_datasets5v_ntk, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader_domain2_ntk = DataLoader(lm_datasets6v_ntk, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class MoMoE_MIXED_LEGAL_REVIEW():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        

        lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)
        
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(800000))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6["train"], range(800000))


        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(500))

        print(len(lm_datasets6t),len(lm_datasets6v),len(lm_datasets5t),len(lm_datasets5v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets6t,lm_datasets5t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets6v,lm_datasets5v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class MoMoE_MIXED_0429():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/reviewforLM", str(self.block_size))
        path7 = os.path.join('/home/jxzhou/datasets/wikipediaforLM', str(self.block_size))
        
        lm_datasets7 = load_from_disk(path7)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)
        
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(100000))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6["train"], range(100000))


        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(5000))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(5000))

        print(len(lm_datasets6t),len(lm_datasets6v),len(lm_datasets5t),len(lm_datasets5v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets6t,lm_datasets5t,lm_datasets7['train']])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets6v,lm_datasets5v,lm_datasets7['validation']])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class MoMoE_SINGLE():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        # path5 = os.path.join("/home/jxzhou/datasets/PubMedForLM", str(self.block_size))
        

        lm_datasets5 = load_from_disk(path5)

        
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(800000))



        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))
      

        print(len(lm_datasets5t),len(lm_datasets5v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets5t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets5v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)




class MoMoE_FEWER_SPECIFIC_GENERAL():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        
        
        path4 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/PubMedForLM", str(self.block_size))
        
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)

        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(800000))
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(100000))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6["train"], range(100000))

        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(500))
        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(500))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(500))

        print(len(lm_datasets4t),len(lm_datasets4v),len(lm_datasets6t),len(lm_datasets6v),len(lm_datasets5t),len(lm_datasets5v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets6t,lm_datasets5t,lm_datasets4t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets6v,lm_datasets5v,lm_datasets4v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class MoMoE_LARGE_SPECIFIC_GENERAL():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        
        
        path4 = os.path.join("/home/jxzhou/datasets/wikitxt103forLM", str(self.block_size))
        path5 = os.path.join("/home/jxzhou/datasets/legalforLM", str(self.block_size))
        path6 = os.path.join("/home/jxzhou/datasets/PubMedForLM", str(self.block_size))
        
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)

        lm_datasets4t = torch.utils.data.Subset(lm_datasets4['train'], range(800000))
        lm_datasets5t = torch.utils.data.Subset(lm_datasets5['train'], range(800000))
        lm_datasets6t = torch.utils.data.Subset(lm_datasets6["train"], range(800000))

        lm_datasets4v = torch.utils.data.Subset(lm_datasets4['validation'], range(1500))
        lm_datasets5v = torch.utils.data.Subset(lm_datasets5['validation'], range(1500))
        lm_datasets6v = torch.utils.data.Subset(lm_datasets6['validation'], range(1500))

        print(len(lm_datasets4t),len(lm_datasets4v),len(lm_datasets6t),len(lm_datasets6v),len(lm_datasets5t),len(lm_datasets5v))
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets6t,lm_datasets5t,lm_datasets4t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets6v,lm_datasets5v,lm_datasets4v])
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)


class GAD():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        data_files = {'train': [f'/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/TASK/REdata/GAD/{i}/train2.csv' for i in range(1,11)],'test': [f'/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/TASK/REdata/GAD/{i}/test2.csv' for i in range(1,11)]}
        datasets = load_dataset("csv",data_files=data_files)
        # print(datasets['train'][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['sentence'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["sentence","index"])
        # print(len(tokenized_datasets['train']),len(tokenized_datasets['test']))
        tokenized_datasets.set_format("torch")
        
        # print(tokenized_datasets['train']['label'])
        # tokenized_datasets.save_to_disk('/home/jxzhou/PLM_PER/finetunedatasets/GAD')


        # lm_datasets5t = torch.utils.data.Subset(tokenized_datasets['train'], range(5000))
        # lm_datasets5v = torch.utils.data.Subset(tokenized_datasets['test'], range(2000))


        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=True)
class GAD_w():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        data_files = {'train': [f'/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/TASK/REdata/GAD/{i}/train2.csv' for i in range(1,11)],'test': [f'/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/TASK/REdata/GAD/{i}/test2.csv' for i in range(1,11)]}
        datasets = load_dataset("csv",data_files=data_files)
        # print(datasets['train'][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['sentence'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["sentence","index"])
        # print(len(tokenized_datasets['train']),len(tokenized_datasets['test']))


        W_ids1 = torch.load('0416-W_IDS-gad-T.pth')
        W_ids2 = torch.load('0416-W_IDS-gad-V.pth')
        def add_index_column_train(example,index):

            example["w_ids"] =  W_ids1[index]

            return example
        def add_index_column_val(example,index):

            example["w_ids"] =  W_ids2[index]

            return example
        
        
        tokenized_datasets['train'] =tokenized_datasets['train'].map(add_index_column_train, with_indices=True)
        tokenized_datasets['test'] =tokenized_datasets['test'].map(add_index_column_val, with_indices=True)
        tokenized_datasets.set_format("torch")
        
        # print(tokenized_datasets['train']['label'])
        tokenized_datasets.save_to_disk('/home/jxzhou/PLM_PER/finetunedatasets/GAD_w')


        # lm_datasets5t = torch.utils.data.Subset(tokenized_datasets['train'], range(5000))
        # lm_datasets5v = torch.utils.data.Subset(tokenized_datasets['test'], range(2000))


        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=True)



class EUADR():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        data_files = {'train': '/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/TASK/REdata/euadr/train.tsv' ,'test': '/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/TASK/REdata/euadr/test.tsv'}
        datasets = load_dataset("csv",data_files=data_files,delimiter='\t')
        # print(datasets['train'][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['sentence'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["sentence"])
        # print(len(tokenized_datasets['train']),len(tokenized_datasets['test']))
        
        
        tokenized_datasets.set_format("torch")
        print(len(tokenized_datasets['train']),len(tokenized_datasets['test']))
        
        # print(tokenized_datasets['train']['label'])
        # tokenized_datasets.save_to_disk('/home/jxzhou/PLM_PER/finetunedatasets/EUADR')


        # lm_datasets5t = torch.utils.data.Subset(tokenized_datasets['train'], range(5000))
        # lm_datasets5v = torch.utils.data.Subset(tokenized_datasets['test'], range(2000))
        # tokenized_datasets['train'] = torch.utils.data.Subset(tokenized_datasets['train'], range(2000))


        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=True)

class EUADR_w():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        data_files = {'train': '/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/TASK/REdata/euadr/train.tsv' ,'test': '/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/TASK/REdata/euadr/test.tsv'}
        datasets = load_dataset("csv",data_files=data_files,delimiter='\t')
        # print(datasets['train'][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['sentence'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["sentence"])
        # print(len(tokenized_datasets['train']),len(tokenized_datasets['test']))
        
        W_ids1 = torch.load('0416-W_IDS-euadr-T.pth')
        W_ids2 = torch.load('0416-W_IDS-euadr-V.pth')
        def add_index_column_train(example,index):

            example["w_ids"] =  W_ids1[index]

            return example
        def add_index_column_val(example,index):

            example["w_ids"] =  W_ids2[index]

            return example
        
        
        tokenized_datasets['train'] =tokenized_datasets['train'].map(add_index_column_train, with_indices=True)
        tokenized_datasets['test'] =tokenized_datasets['test'].map(add_index_column_val, with_indices=True)
        tokenized_datasets.set_format("torch")
        
        # print(tokenized_datasets['train']['label'])
        tokenized_datasets.save_to_disk('/home/jxzhou/PLM_PER/finetunedatasets/EUADR_w')


        tokenized_datasets['train'] = torch.utils.data.Subset(tokenized_datasets['train'], range(2000))
        # lm_datasets5v = torch.utils.data.Subset(tokenized_datasets['test'], range(2000))


        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=True)




class GAD_single():
    def __init__(self, config,n) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        data_files = {'train': f'/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/TASK/REdata/GAD/{n}/train2.csv','test': f'/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/TASK/REdata/GAD/{n}/test2.csv'}
        datasets = load_dataset("csv",data_files=data_files)
        # print(datasets['train'][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['sentence'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["sentence","index"])
        print(len(tokenized_datasets['train']),len(tokenized_datasets['test']))
        tokenized_datasets.set_format("torch")
        
        print(tokenized_datasets['train'][0])

        # lm_datasets5t = torch.utils.data.Subset(tokenized_datasets['train'], range(5000))
        # lm_datasets5v = torch.utils.data.Subset(tokenized_datasets['test'], range(2000))


        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)



class Overruling():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        data_files = {'train':'/home/jxzhou/PLM_PER/MoMoE_rawdata/LAW/TASK/overruling.csv'}
        datasets = load_dataset("csv",data_files=data_files)
        datasets["test"] = load_dataset(
            'csv', data_files=data_files,split=f"train[:{5}%]"
        )
        datasets["train"] = load_dataset(
            'csv', data_files=data_files,
            split=f"train[{5}%:]",
        )
        # print(datasets['train'][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['sentence1'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["sentence1"])
        tokenized_datasets.set_format("torch")
        # tokenized_datasets.save_to_disk('/home/jxzhou/PLM_PER/finetunedatasets/Overruling')
        
        # print(len(tokenized_datasets['train']),len(tokenized_datasets['test']))
        # print(tokenized_datasets['train']['label'])

        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=True)


class Overruling_w():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        data_files = {'train':'/home/jxzhou/PLM_PER/MoMoE_rawdata/LAW/TASK/overruling.csv'}
        datasets = load_dataset("csv",data_files=data_files)
        datasets["test"] = load_dataset(
            'csv', data_files=data_files,split=f"train[:{5}%]"
        )
        datasets["train"] = load_dataset(
            'csv', data_files=data_files,
            split=f"train[{5}%:]",
        )



        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['sentence1'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["sentence1"])
        # print(datasets['train'][0])
        W_ids1 = torch.load('0416-W_IDS-overruling-T.pth')
        W_ids2 = torch.load('0416-W_IDS-overruling-V.pth')
        def add_index_column_train(example,index):

            example["w_ids"] =  W_ids1[index]

            return example
        def add_index_column_val(example,index):

            example["w_ids"] =  W_ids2[index]

            return example
        
        
        tokenized_datasets['train'] =tokenized_datasets['train'].map(add_index_column_train, with_indices=True)
        tokenized_datasets['test'] =tokenized_datasets['test'].map(add_index_column_val, with_indices=True)
        
        
        tokenized_datasets.set_format("torch")
        tokenized_datasets.save_to_disk('/home/jxzhou/PLM_PER/finetunedatasets/Overruling_w')
        
        # print(len(tokenized_datasets['train']),len(tokenized_datasets['test']))
        # print(tokenized_datasets['train']['label'])

        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=True)


class Law_stack():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        data_files = {'train':'/home/jxzhou/PLM_PER/MoMoE_rawdata/LAW/TASK/law-stack-exchange/renamed_train.jsonl','val':'/home/jxzhou/PLM_PER/MoMoE_rawdata/LAW/TASK/law-stack-exchange/renamed_test.jsonl'}
        datasets = load_dataset("json",data_files=data_files)
        datasets["test"] = load_dataset(
            'json', data_files=data_files,split=f"val"
        )
        datasets["train"] = load_dataset(
            'json', data_files=data_files,
            split="train",
        )
        # print(datasets['train'][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['body'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["body"])
        tokenized_datasets.set_format("torch")
        
        print(len(tokenized_datasets['train']),len(tokenized_datasets['test']))
        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=True)



class medical_abstract():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        data_files = {'train':'/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/medical-abstract/updated_train_v2.csv','val':'/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/medical-abstract/updated_test_v2.csv',"longtailed":"/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/medical-abstract/updated_test_label_2_v2.csv"}
        datasets = load_dataset("csv",data_files=data_files)
        datasets["test"] = load_dataset(
            'csv', data_files=data_files,split="val"
        )
        datasets["train"] = load_dataset(
            'csv', data_files=data_files,
            split="train",
        )
        datasets["longtailed"] = load_dataset(
            'csv', data_files=data_files,
            split="longtailed",
        )
        # print(datasets['train'][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['medical_abstract'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["medical_abstract"])
        tokenized_datasets.set_format("torch")
        tokenized_datasets['train'] = torch.utils.data.Subset(tokenized_datasets['train'], range(1000))
        print(len(tokenized_datasets['train']),len(tokenized_datasets['test']),len(tokenized_datasets['longtailed']))
        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=True)
        self.val_loader_longtailed = DataLoader(tokenized_datasets['longtailed'], batch_size=self.batch_size, shuffle=True)





class legal_argument_mining():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        data_files = {'train':'/home/jxzhou/PLM_PER/MoMoE_rawdata/LAW/TASK/legal_argument_mining/updated_train.csv','val':'/home/jxzhou/PLM_PER/MoMoE_rawdata/LAW/TASK/legal_argument_mining/updated_test.csv','longtailed':"/home/jxzhou/PLM_PER/MoMoE_rawdata/LAW/TASK/legal_argument_mining/test_conclusion_only.csv"}
        datasets = load_dataset("csv",data_files=data_files)
        datasets["test"] = load_dataset(
            'csv', data_files=data_files,split="val"
        )
        datasets["train"] = load_dataset(
            'csv', data_files=data_files,
            split="train",
        )
        datasets["longtailed"] = load_dataset(
            'csv', data_files=data_files,
            split="longtailed",
        )
        # print(datasets['train'][0])

        
        datasets.map(self.convert_labels_to_int64)
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['input_sentence_en'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["input_sentence_en","input_sentence"])
        tokenized_datasets.set_format("torch")
        # tokenized_datasets['train']['label'] = tokenized_datasets['train']['label'].type(torch.cuda.LongTensor)
        # tokenized_datasets['test']['label'] = tokenized_datasets['train']['label'].type(torch.cuda.LongTensor)

        # tokenized_datasets['longtailed']['label'] = tokenized_datasets['train']['label'].type(torch.cuda.LongTensor)

        print(len(tokenized_datasets['train']),len(tokenized_datasets['test']),len(tokenized_datasets['longtailed']))
        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        print(tokenized_datasets['train'][0])
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)
        self.val_loader_longtailed = DataLoader(tokenized_datasets['longtailed'], batch_size=self.batch_size, shuffle=False)
    def convert_labels_to_int64(self,example):
        example['label'] = int(example['label'])
        return example
class Casehold():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        data_files = {'train':'/home/jxzhou/PLM_PER/MoMoE_rawdata/LAW/TASK/casehold_binary_cls_train.csv','test':'/home/jxzhou/PLM_PER/MoMoE_rawdata/LAW/TASK/casehold_binary_cls_test.csv'}
        datasets = load_dataset("csv",data_files=data_files)
        datasets["test"] = load_dataset(
            'csv', data_files=data_files,split="test"
        )
        datasets["train"] = load_dataset(
            'csv', data_files=data_files,
            split="train",
        )
        # print(datasets['train'][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['text'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["text"])
        
        
        tokenized_datasets.set_format("torch")
        
        # print(len(tokenized_datasets['train']),len(tokenized_datasets['test']))
        # print(tokenized_datasets['train']['label'])
        # tokenized_datasets.save_to_disk('/home/jxzhou/PLM_PER/finetunedatasets/Casehold')


        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=True)


class Casehold_w():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        data_files = {'train':'/home/jxzhou/PLM_PER/MoMoE_rawdata/LAW/TASK/casehold_binary_cls_train.csv','test':'/home/jxzhou/PLM_PER/MoMoE_rawdata/LAW/TASK/casehold_binary_cls_test.csv'}
        datasets = load_dataset("csv",data_files=data_files)
        datasets["test"] = load_dataset(
            'csv', data_files=data_files,split="test"
        )
        datasets["train"] = load_dataset(
            'csv', data_files=data_files,
            split="train",
        )
        # print(datasets['train'][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['text'], padding='max_length',max_length=config.seq_len,truncation='longest_first'), batched=True,remove_columns=["text"])
        W_ids1 = torch.load('0416-W_IDS-casehold-T.pth')
        W_ids2 = torch.load('0416-W_IDS-casehold-V.pth')
        print(W_ids1)
        def add_index_column_train(example,index):

            example["w_ids"] =  W_ids1[index]

            return example
        def add_index_column_val(example,index):

            example["w_ids"] =  W_ids2[index]

            return example
        
        
        tokenized_datasets['train'] =tokenized_datasets['train'].map(add_index_column_train, with_indices=True)
        tokenized_datasets['test'] =tokenized_datasets['test'].map(add_index_column_val, with_indices=True)
        
        tokenized_datasets.set_format("torch")
        
        # print(len(tokenized_datasets['train']),len(tokenized_datasets['test']))
        # print(tokenized_datasets['train'][0]['label'])
        tokenized_datasets.save_to_disk('/home/jxzhou/PLM_PER/finetunedatasets/Casehold_w')


        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        # self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=True)



class Wikipedia():
    def group_texts(self, examples):
        block_size = self.block_size

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        num_proc = multiprocessing.cpu_count() // 2
        data_files = {'train': '/home/jxzhou/PLM_PER/wikipedia-0330'}
        # raw_datasets = load_dataset('/home/jxzhou/PLM_PER/wikipedia-0330')
        raw_datasets = load_dataset('/home/jxzhou/PLM_PER/wikipedia-0330', split=[f"train[:{99}%]", f"train[{99}%:]"])
        raw_datasets = DatasetDict({name: dataset for name, dataset in zip(['train','validation'], raw_datasets)})
        # raw_datasets["validation"] = load_dataset(
        #     '/home/jxzhou/PLM_PER/wikipedia-0330',split=f"train[:{1}%]"
        # )
        # raw_datasets["train"] = load_dataset(
        #     '/home/jxzhou/PLM_PER/wikipedia-0330',
        #     split=f"train[{99}%:]",
        # )

        # tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text']), batched=True, num_proc=num_proc, remove_columns=["text"])
        # lm_dataset = tokenized_datasets.map(self.group_texts, batched=True)



        column_names = raw_datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = raw_datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=['id','url','title',"text"],
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        # lm_dataset.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path = os.path.join('/home/jxzhou/datasets/wikipediaforLM', str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        print(len(lm_datasets['train']),len(lm_datasets['validation']))
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
class Wikipedia_small():
    def group_texts(self, examples):
        block_size = self.block_size

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        num_proc = multiprocessing.cpu_count() // 2
        data_files = {'train': '/home/jxzhou/PLM_PER/wikipedia-0330'}
        # raw_datasets = load_dataset('/home/jxzhou/PLM_PER/wikipedia-0330')
        raw_datasets = load_dataset('/home/jxzhou/PLM_PER/wikipedia-0330', split=[f"train[:{99}%]", f"train[{99}%:]"])
        raw_datasets = DatasetDict({name: dataset for name, dataset in zip(['train','validation'], raw_datasets)})
        # raw_datasets["validation"] = load_dataset(
        #     '/home/jxzhou/PLM_PER/wikipedia-0330',split=f"train[:{1}%]"
        # )
        # raw_datasets["train"] = load_dataset(
        #     '/home/jxzhou/PLM_PER/wikipedia-0330',
        #     split=f"train[{99}%:]",
        # )

        # tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text']), batched=True, num_proc=num_proc, remove_columns=["text"])
        # lm_dataset = tokenized_datasets.map(self.group_texts, batched=True)



        column_names = raw_datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = raw_datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=['id','url','title',"text"],
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        # lm_dataset.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config,KKK):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path = os.path.join('/home/jxzhou/datasets/wikipediaforLM', str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        lm_datasets['train'] = lm_datasets['train'].select(range(KKK))
        lm_datasets['validation'] = lm_datasets['validation'].select(range(1000))
        print(len(lm_datasets['train']),len(lm_datasets['validation']))
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)




class Wikipedia_forids():
    def group_texts(self, examples):
        block_size = self.block_size

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        num_proc = multiprocessing.cpu_count() // 2
        data_files = {'train': '/home/jxzhou/PLM_PER/wikipedia-0330'}
        # raw_datasets = load_dataset('/home/jxzhou/PLM_PER/wikipedia-0330')
        raw_datasets = load_dataset('/home/jxzhou/PLM_PER/wikipedia-0330', split=[f"train[:{99}%]", f"train[{99}%:]"])
        raw_datasets = DatasetDict({name: dataset for name, dataset in zip(['train','validation'], raw_datasets)})
        # raw_datasets["validation"] = load_dataset(
        #     '/home/jxzhou/PLM_PER/wikipedia-0330',split=f"train[:{1}%]"
        # )
        # raw_datasets["train"] = load_dataset(
        #     '/home/jxzhou/PLM_PER/wikipedia-0330',
        #     split=f"train[{99}%:]",
        # )

        # tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text']), batched=True, num_proc=num_proc, remove_columns=["text"])
        # lm_dataset = tokenized_datasets.map(self.group_texts, batched=True)



        column_names = raw_datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = raw_datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=['id','url','title',"text"],
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )
        tokenized_datasets.save_to_disk(path)
        # lm_dataset.save_to_disk(path)
        return tokenized_datasets

    def __init__(self, config,t):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path = os.path.join('/home/jxzhou/datasets/wikipediaforLM', str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        lim =  (t+1)*4520000 < len(lm_datasets['train']) and (t+1)*4520000 or len(lm_datasets['train'])
        print(lim)
        lm_datasets['train'] = torch.utils.data.Subset(lm_datasets['train'], range(t*4520000,lim))
        # lm_datasets['validation'] = torch.utils.data.Subset(lm_datasets['validation'], range(t*11300,lim))
        print(len(lm_datasets['train']),len(lm_datasets['validation']))
        # print(lm_datasets['train'][0])
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class SST2():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        # self.task_name = config.task_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        # datasets = load_dataset("glue", self.task_name)
        datasets = load_from_disk('/home/jxzhou/datasets/glue/sst2')
        # print(11,datasets["train"][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['sentence'], padding='max_length', max_length=config.seq_len,truncation='longest_first'), batched=True, remove_columns=["sentence", "idx"])

        tokenized_datasets.set_format("torch")
        tokenized_datasets.save_to_disk('/home/jxzhou/PLM_PER/finetunedatasets/SST2')
        print(len(tokenized_datasets['train']),len(tokenized_datasets['validation']))
        # print(tokenized_datasets['train']['label'])
        
        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=True)
        # self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)


class COLA():
    def __init__(self, config) -> None:
        # self.model_name = config.model_name
        # self.task_name = config.task_name
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        # datasets = load_dataset("glue", self.task_name)
        datasets = load_from_disk('/home/jxzhou/datasets/glue/cola')
        # print(11,datasets["train"][0])
        tokenized_datasets = datasets.map(lambda dataset: self.tokenizer(dataset['sentence'], padding='max_length', max_length=config.seq_len,truncation='longest_first'), batched=True, remove_columns=["sentence", "idx"])

        tokenized_datasets.set_format("torch")
        # tokenized_datasets.save_to_disk('/home/jxzhou/PLM_PER/finetunedatasets/cola')
        print(len(tokenized_datasets['train']),len(tokenized_datasets['validation']))
        # print(tokenized_datasets['train']['label'])
        
        self.train_loader = DataLoader(tokenized_datasets['train'], batch_size=self.batch_size, shuffle=True)
        
        
        self.val_loader = DataLoader(tokenized_datasets['validation'], batch_size=self.batch_size, shuffle=True)
        # self.val_loader = DataLoader(tokenized_datasets['test'], batch_size=self.batch_size, shuffle=False)


class multi_domains():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        
        
        path4 = os.path.join("/home/jxzhou/PLM_PER/finetunedatasets/Casehold")
        path5 = os.path.join("/home/jxzhou/PLM_PER/finetunedatasets/GAD")
        path6 = os.path.join("/home/jxzhou/PLM_PER/finetunedatasets/Overruling")
        path7 = os.path.join("/home/jxzhou/PLM_PER/finetunedatasets/SST2")
        
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)
        lm_datasets7 = load_from_disk(path7)

        lm_datasets4t = lm_datasets4['train']
        lm_datasets5t = lm_datasets5['train']
        lm_datasets6t = lm_datasets6["train"]
        lm_datasets7t = lm_datasets7["train"]

        lm_datasets4v = lm_datasets4['test']
        lm_datasets5v = lm_datasets5['test']
        lm_datasets6v = lm_datasets6['test']
        lm_datasets7v = lm_datasets7['validation']

        
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets7t,lm_datasets6t,lm_datasets5t,lm_datasets4t])
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets7v,lm_datasets6v,lm_datasets5v,lm_datasets4v])
        print(len(lm_datasets),len(lm_datasets_val))
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader1 = DataLoader(lm_datasets4v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader2 = DataLoader(lm_datasets5v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader3 = DataLoader(lm_datasets6v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader4 = DataLoader(lm_datasets7v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader1 = DataLoader(lm_datasets4t, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.train_loader2 = DataLoader(lm_datasets5t, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.train_loader3 = DataLoader(lm_datasets6t, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.train_loader4 = DataLoader(lm_datasets7t, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        # self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)

class multi_domains_wids():
    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')
        
        
        path4 = os.path.join("/home/jxzhou/PLM_PER/finetunedatasets/Casehold")
        path5 = os.path.join("/home/jxzhou/PLM_PER/finetunedatasets/GAD")
        path6 = os.path.join("/home/jxzhou/PLM_PER/finetunedatasets/Overruling")
        path7 = os.path.join("/home/jxzhou/PLM_PER/finetunedatasets/SST2")
        
        lm_datasets4 = load_from_disk(path4)
        lm_datasets5 = load_from_disk(path5)
        lm_datasets6 = load_from_disk(path6)
        lm_datasets7 = load_from_disk(path7)

        lm_datasets4t = lm_datasets4['train']
        lm_datasets5t = lm_datasets5['train']
        lm_datasets6t = lm_datasets6["train"]
        lm_datasets7t = lm_datasets7["train"]

        lm_datasets4v = lm_datasets4['test']
        lm_datasets5v = lm_datasets5['test']
        lm_datasets6v = lm_datasets6['test']
        lm_datasets7v = lm_datasets7['validation']

        W_ids = torch.load('0410-W_IDS-FINETUNETRAIN.pth')
        W_ids1 = torch.load('0410-W_IDS-FINETUNEVAL.pth')
        W_ids2 = torch.load('0410-W_IDS-FINETUNETRAINS.pth')

        def add_index_column_train(example,index):
            example["w_ids"] =  W_ids[index]

            return example
        def add_index_column_train1(example,index):

            example["w_ids"] =  W_ids2[0,index]

            return example
        def add_index_column_train2(example,index):

            example["w_ids"] =  W_ids2[1,index]

            return example
        def add_index_column_train3(example,index):

            example["w_ids"] =  W_ids2[2,index]

            return example
        def add_index_column_train4(example,index):

            example["w_ids"] =  W_ids2[3,index]

            return example
        def add_index_column_val1(example,index):

            example["w_ids"] =  W_ids1[0,index]

            return example
        def add_index_column_val2(example,index):

            example["w_ids"] =  W_ids1[1,index]

            return example
        def add_index_column_val3(example,index):

            example["w_ids"] =  W_ids1[2,index]

            return example
        def add_index_column_val4(example,index):

            example["w_ids"] =  W_ids1[3,index]

            return example
        lm_datasets4v =lm_datasets4v.map(add_index_column_val1, with_indices=True)
        lm_datasets5v =lm_datasets5v.map(add_index_column_val2, with_indices=True)
        lm_datasets6v =lm_datasets6v.map(add_index_column_val3, with_indices=True)
        lm_datasets7v =lm_datasets7v.map(add_index_column_val4, with_indices=True)
        
        lm_datasets4t =lm_datasets4t.map(add_index_column_train1, with_indices=True)
        lm_datasets5t =lm_datasets5t.map(add_index_column_train2, with_indices=True)
        lm_datasets6t =lm_datasets6t.map(add_index_column_train3, with_indices=True)
        lm_datasets7t =lm_datasets7t.map(add_index_column_train4, with_indices=True)
        lm_datasets = torch.utils.data.ConcatDataset([lm_datasets7t,lm_datasets6t,lm_datasets5t,lm_datasets4t])
        
        # lm_datasets =lm_datasets.map(add_index_column_train, with_indices=True)
        
        lm_datasets_val = torch.utils.data.ConcatDataset([lm_datasets7v,lm_datasets6v,lm_datasets5v,lm_datasets4v])
        
        
        print(len(lm_datasets),len(lm_datasets_val))
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets_val, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader1 = DataLoader(lm_datasets4v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader2 = DataLoader(lm_datasets5v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader3 = DataLoader(lm_datasets6v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_loader4 = DataLoader(lm_datasets7v, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        

class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        # 提取索引列并从数据中移除
        w_ids = torch.tensor([example["w_ids"] for example in examples])
        # print(w_ids)
        examples_without_indices = [{k: v for k, v in example.items() if k != "w_ids"} for example in examples]
        
        # 使用父类方法处理没有索引的文本数据
        batch = super().__call__(examples_without_indices)
        
        # 将索引信息添加回批处理的数据中
        # 假设我们以某种形式将索引添加为一个额外的键值对
        batch['w_ids'] = w_ids
        return batch
class Wikipedia_Wids():
    def group_texts(self, examples):
        block_size = self.block_size

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True)
    
    def preprocess(self, config, path):
        num_proc = multiprocessing.cpu_count() // 2
        data_files = {'train': '/home/jxzhou/PLM_PER/wikipedia-0330'}
        # raw_datasets = load_dataset('/home/jxzhou/PLM_PER/wikipedia-0330')
        raw_datasets = load_dataset('/home/jxzhou/PLM_PER/wikipedia-0330', split=[f"train[:{99}%]", f"train[{99}%:]"])
        raw_datasets = DatasetDict({name: dataset for name, dataset in zip(['train','validation'], raw_datasets)})
        # raw_datasets["validation"] = load_dataset(
        #     '/home/jxzhou/PLM_PER/wikipedia-0330',split=f"train[:{1}%]"
        # )
        # raw_datasets["train"] = load_dataset(
        #     '/home/jxzhou/PLM_PER/wikipedia-0330',
        #     split=f"train[{99}%:]",
        # )

        # tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text']), batched=True, num_proc=num_proc, remove_columns=["text"])
        # lm_dataset = tokenized_datasets.map(self.group_texts, batched=True)



        column_names = raw_datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]
        tokenized_datasets = raw_datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=['id','url','title',"text"],
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            # load_from_cache_file=not config.overwrite_cache,
            desc=f"Grouping texts in chunks of {1024}",
        )

        # W_ids = torch.zeros(36120193).long()
        # for i in range(8):
        #     W_ids += torch.load('0403-W_IDS-TRAIN%d.pth'%i)
        # # print(W_ids>7)
        # W_ids2 = torch.load('0403-W_IDS-VAL.pth')
        # c = 0
        # d = 0
        # def add_index_column_train(example, index):
        #     c = index
        #     example["w_ids"] =  W_ids[c]
        #     return example
        # def add_index_column_val(example, index):
        #     c = index
        #     # print(index)
        #     example["w_ids"] =  W_ids2[c]
        #     return example
        # tokenized_datasets['train'] =tokenized_datasets['train'].map(add_index_column_train, with_indices=True)
        # tokenized_datasets['validation'] = tokenized_datasets['validation'].map(add_index_column_val, with_indices=True)

        tokenized_datasets.save_to_disk(path)
        # lm_dataset.save_to_disk(path)
        return tokenized_datasets


    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path = os.path.join('/home/jxzhou/datasets/wikipediaforLM/w_ids3')
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        # lm_datasets['train'] = lm_datasets['train'].select(range(1000000))
        # lm_datasets['validation'] = lm_datasets['validation'].select(range(1000))
        
        # W_ids = torch.zeros(36160000).long()

        # for i in range(8):
        #     d = torch.load('0411-W_IDS-WIKI-T%d.pth'%i)
        #     # print(i)
        #     W_ids[4520000*i:4520000*(i+1)]+=d


        # W_ids2 = torch.load('0411-W_IDS-WIKI-V.pth')
        # # W_ids = W_ids.tolist()

        # def add_index_column_train(example,index):
        #     example["w_ids"] =  W_ids[index]

        #     return example
        # def add_index_column_val(example,index):

        #     example["w_ids"] =  W_ids2[index]

        #     return example
        # print('w_ids adding begin')
        # lm_datasets['train'] =lm_datasets['train'].map(add_index_column_train, with_indices=True)
        # lm_datasets['validation'] = lm_datasets['validation'].map(add_index_column_val, with_indices=True)


        # lm_datasets.save_to_disk('/home/jxzhou/datasets/wikipediaforLM/w_ids2')

        print(len(lm_datasets['train']),len(lm_datasets['validation']))
        # print(lm_datasets['train'][10000:13000]["w_ids"])
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=True , collate_fn=data_collator)
        # self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)



class QNLI():
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True,max_length=config.seq_len,truncation='longest_first')
    
    def preprocess_function(self, examples):
        examples[self.text_column_name] = [q + " " + s for q, s in zip(examples['question'], examples['sentence'])]
        # examples["text"] = examples["question"] + " " + examples["sentence"]
        return examples
    def preprocess(self, config, path):

        
        raw_datasets = load_dataset('/home/jxzhou/PLM_PER/BERT-CL-main/GLUE_RAW_DATA/GLUE/GNLI', split=["train", 'validation',"test"])
        raw_datasets = DatasetDict({name: dataset for name, dataset in zip(['train','validation','test'], raw_datasets)})
        raw_datasets = raw_datasets.map(self.preprocess_function, batched=True)

        tokenized_datasets = raw_datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=['question', 'sentence', 'idx',self.text_column_name],
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
            
        )
        tokenized_datasets.save_to_disk(path)

        return tokenized_datasets

    def __init__(self, config):
        self.text_column_name = "text"
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path = os.path.join('/home/jxzhou/datasets/glue/gnli')
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        print(len(lm_datasets['train']),len(lm_datasets['validation']))
        print(lm_datasets['train'][0])
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)

class MRPC():
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True,max_length=config.seq_len,truncation='longest_first')
    
    def preprocess_function(self, examples):
        examples[self.text_column_name] = [q + " " + s for q, s in zip(examples['sentence1'], examples['sentence2'])]
        # examples["text"] = examples["question"] + " " + examples["sentence"]
        return examples
    def preprocess(self, config, path):

        
        raw_datasets = load_dataset('/home/jxzhou/PLM_PER/BERT-CL-main/GLUE_RAW_DATA/GLUE/MRPC', split=["train", 'validation',"test"])
        raw_datasets = DatasetDict({name: dataset for name, dataset in zip(['train','validation','test'], raw_datasets)})
        raw_datasets = raw_datasets.map(self.preprocess_function, batched=True)

        tokenized_datasets = raw_datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=['sentence1', 'sentence2', 'idx',self.text_column_name],
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
            
        )
        tokenized_datasets.save_to_disk(path)

        return tokenized_datasets

    def __init__(self, config):
        self.text_column_name = "text"
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path = os.path.join('/home/jxzhou/datasets/glue/mrpc')
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        print(len(lm_datasets['train']),len(lm_datasets['validation']))
        print(lm_datasets['train'][0])
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)


class QQP():
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True,padding='max_length',max_length=config.seq_len,truncation='longest_first')
    
    def preprocess_function(self, examples):
        examples[self.text_column_name] = [q + " " + s for q, s in zip(examples['question1'], examples['question2'])]
        # examples["text"] = examples["question"] + " " + examples["sentence"]
        return examples
    def preprocess(self, config, path):

        
        raw_datasets = load_dataset('/home/jxzhou/PLM_PER/BERT-CL-main/GLUE_RAW_DATA/GLUE/QQP', split=["train", 'validation',"test"])
        raw_datasets = DatasetDict({name: dataset for name, dataset in zip(['train','validation','test'], raw_datasets)})
        raw_datasets = raw_datasets.map(self.preprocess_function, batched=True)

        tokenized_datasets = raw_datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=['question1', 'question2', 'idx',self.text_column_name],
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
            
        )
        tokenized_datasets.save_to_disk(path)

        return tokenized_datasets

    def __init__(self, config):
        self.text_column_name = "text"
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path = os.path.join('/home/jxzhou/datasets/glue/qqp')
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        print(len(lm_datasets['train']),len(lm_datasets['validation']))
        print(lm_datasets['train'][0])
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)

class RTE():
    def tokenize_function(self, examples):
        return self.tokenizer(examples[self.text_column_name], return_special_tokens_mask=True,padding='max_length',max_length=config.seq_len,truncation='longest_first')
    
    def preprocess_function(self, examples):
        examples[self.text_column_name] = [q + " " + s for q, s in zip(examples['sentence1'], examples['sentence2'])]
        # examples["text"] = examples["question"] + " " + examples["sentence"]
        return examples
    def preprocess(self, config, path):

        
        raw_datasets = load_dataset('/home/jxzhou/PLM_PER/BERT-CL-main/GLUE_RAW_DATA/GLUE/RTE', split=["train", 'validation',"test"])
        raw_datasets = DatasetDict({name: dataset for name, dataset in zip(['train','validation','test'], raw_datasets)})
        raw_datasets = raw_datasets.map(self.preprocess_function, batched=True)

        tokenized_datasets = raw_datasets.map(
            self.tokenize_function,
            batched=True,
            # num_proc=config.preprocessing_num_workers,
            remove_columns=['sentence1', 'sentence2', 'idx',self.text_column_name],
            # load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets.save_to_disk(path)

        return tokenized_datasets

    def __init__(self, config):
        self.text_column_name = "text"
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/jxzhou/PLM_PER/MODELS/roberta-base')

        path = os.path.join('/home/jxzhou/datasets/glue/rte')
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        print(len(lm_datasets['train']),len(lm_datasets['validation']))
        print(lm_datasets['train'][0])
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)



if __name__ == "__main__":
    config = BertConfig.from_json_file('config/datasets.json')
    # dataset =EUADR_w(config)
    dataset = RTE(config)
    dataset = QQP(config)
    dataset = MRPC(config)
    dataset = QNLI(config)

    # textloader = dataset.train_loader
    # for i, batch in enumerate(textloader):
    #     if i>=100:
    #         print(batch['w_ids'])
    #         break
    



