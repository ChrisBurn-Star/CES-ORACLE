from cProfile import label
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from utils import Benchmark_ultrachat_FewShot_Qwen3
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import re
from datasets import Dataset

import os
import json
import torch
import torch.nn as nn

from utils import DeepSeekDistillation

from accelerate import infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from collections import Counter


from transformers import LogitsProcessor, LogitsProcessorList
import os
import torch
import torch.nn as nn

from utils import DeepSeekDistillation
from models import MyQwen3ForCausalLM # 这里还需要 models/myqwen.py, qwen3config.py 文件
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup, AutoModelForCausalLM
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
# from transformer_engine.pytorch import fp8_autocast
from accelerate import dispatch_model
from accelerate import Accelerator
import re
import string
from collections import Counter

import numpy as np
from tqdm import tqdm



new_tokens = 300


def benchmarktest(test_data, model,tokenizer):

    dataloader = DataLoader(test_data, batch_size=1, num_workers=4)
    for batch in tqdm(dataloader):
        model_inputs = batch['model_inputs']
        input_len = model_inputs["input_ids"].squeeze(0).shape[1]

        with torch.no_grad():
            start_time = time.time()
            output = model(input_ids = model_inputs["input_ids"].squeeze(0).to(model.device), output_hidden_states=True, output_attentions=False, labels = model_inputs["input_ids"].squeeze(0).to(model.device),output_expert_labels = False )
            stop_time = time.time()
            times = stop_time-start_time
            
            
            
            print(f"inference time: {times:.4f} s/ {input_len} tokens")
            
            s = time.time()
            output_ids = model.generate(
                input_ids=model_inputs["input_ids"].squeeze(0).to(model.device),
                attention_mask=model_inputs["attention_mask"].squeeze(0).to(model.device),
                max_new_tokens=new_tokens,
                no_repeat_ngram_size=3,   # 防止重复
                do_sample=True,           # 启用采样生成更自然的回答
                top_k=10,                # Top-k 采样
                temperature=0.2,          # 控制生成多样性
            )
            t = time.time()

            print(f"generate time: {t-s}s/ ({input_len} tokens + {new_tokens} tokens)")







    
    
    

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/project/yunweiyuhuifu/p-shangli/Qwen3-8B", trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained('/inspire/hdd/project/yunweiyuhuifu/p-shangli/Qwen3-8B', device_map="cuda:0", trust_remote_code=True)

    test_data = Benchmark_ultrachat_FewShot_Qwen3(f'/home/jxzhou/PLM_PER/datasets', tokenizer)
    benchmarktest(test_data, qwen,tokenizer)
