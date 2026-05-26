from cProfile import label
from tkinter import Label
from seaborn import heatmap
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from old_data_utils import Tokenized_data, Tokenized_data_chat, Tokenized_data_locality
from sklearn.decomposition import PCA
import umap
from sklearn.metrics.pairwise import cosine_similarity
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from utils import Benchmark_ultrachat_FewShot_Qwen3,Benchmark_Mag_FewShot_Qwen3,Benchmark_TriviaQA_FewShot_Qwen3,Benchmark_XSUM_FewShot_Qwen3,Benchmark_GPQA_Qwen3,Benchmark_DROP_FewShot_Qwen3,compute_exact,compute_f1,Benchmark_BBH_FewShot_Qwen3,Benchmark_MMLU_FewShot_Qwen3,Benchmark_DROP, Benchmark_MMLU, Benchmark_GPQA,Benchmark_BBH,Benchmark_MMLU_FewShot,Benchmark_BBH_FewShot,Benchmark_XSUM_FewShot
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import re
from datasets import Dataset

import os
import json
import torch
import torch.nn as nn

from models import MyDeepseekV2MoE
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

from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm

def rouge1_recall(predicted: str, reference: str) -> float:
    """
    计算 ROUGE-1 召回率（Recall）：参考答案中的词有多少比例出现在生成的答案中。

    参数：
    predicted: 模型输出字符串
    reference: 标准答案字符串

    返回：
    float: ROUGE-1 召回率（0~1之间）
    """

    def clean_text(text: str) -> str:
        # 去除特殊标签和标点，归一化空白符
        text = re.sub(r'</think>', ' ', text)
        text = re.sub(r'["“”]', '', text)
        text = re.sub(r'\s+', ' ', text)  # 把多个空格/换行变成一个空格
        return text.strip().lower()

    # 清洗并分词
    pred_tokens = clean_text(predicted).split()
    ref_tokens = clean_text(reference).split()

    # 统计词频
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # 计算匹配的词个数
    match_count = sum(min(ref_counter[word], pred_counter.get(word, 0)) for word in ref_counter)

    # 计算召回率
    recall = match_count / len(ref_tokens) if ref_tokens else 0.0
    return recall








def normalize_answer(s):
    import re, string
    return ''.join(ch for ch in s.lower() if ch not in string.punctuation).strip()

def triviaqa_match(prediction, answers):
    pred_norm = normalize_answer(prediction)
    for ans in answers:
        if normalize_answer(ans) in pred_norm:
            return True
    return False
def compute_rouge_per_sample(predictions, references):
    """
    针对每一组预测和参考摘要，单独计算 ROUGE 分数。
    
    Args:
        predictions: list of str，模型生成的摘要
        references: list of str，对应的参考摘要（标准答案）
        
    Returns:
        results: list of dict，每个样本对应一个 dict，含 ROUGE-1, 2, L 的 f-score
    """

    

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    score = scorer.score(references, predictions)
    results = {'rouge1': round(score['rouge1'].fmeasure * 100, 2),
            'rouge2': round(score['rouge2'].fmeasure * 100, 2),
            'rougeL': round(score['rougeL'].fmeasure * 100, 2)}
    
    
    return results
def normalize_answer(s):
    """去除文章中的标点、大小写、前后空格等规范化操作"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_answer(s):
    """标准化答案：小写、去标点、去冠词、去空格"""
    def lower(text):
        return text.lower()
    
    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)



class ForceChoiceProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids, step=0):
        self.allowed_token_ids = set(allowed_token_ids)
        self.step = step  # 只作用于第几步生成（例如 0 表示第一个token）

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] == self.step + 1:  # 只在第 step 步做筛选
            mask = torch.full_like(scores, float("-inf"))
            for idx in self.allowed_token_ids:
                mask[:, idx] = scores[:, idx]
            return mask
        return scores


# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'


# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,7'

BATCH_SIZE = 8
SEQ_LEN = 2048
EXPERT_NUM = 64
TOPK_EXPERT = 6
T_DTYPE = torch.bfloat16
S_DTYPE = torch.float32
GATING_REFERENCE = 'attn_output'
LLM_DIR = "/home/jxzhou/PLM_PER/qwen/DeepSeek-16B-2.8B"


TASK = 'ULTRACHAT'
NEW_TOKENS = 300  ####MMLU 3; BBH 300 XSUM 30
PAD = 5 #####MMLU:5 BBH:5 XSUM:0
keyword = "Answer:"  ###MMLU
# keyword = "!"  ###MMLU

# keyword = "So the answer is" ### BBH
# keyword = "Summary"  ###XSUM
DATASET = {'MMLU': Benchmark_MMLU_FewShot_Qwen3,'BBH':Benchmark_BBH_FewShot_Qwen3, 'XSUM/processed':Benchmark_XSUM_FewShot, 'DROP':Benchmark_DROP_FewShot_Qwen3, 'GPQA':Benchmark_GPQA_Qwen3,'XSUM':Benchmark_XSUM_FewShot_Qwen3,'TRIVIAQA':Benchmark_TriviaQA_FewShot_Qwen3,'MAG':Benchmark_Mag_FewShot_Qwen3, 'ULTRACHAT':Benchmark_ultrachat_FewShot_Qwen3}

if TASK == 'MMLU' or TASK == 'BBH':
    base_dir = f'/home/jxzhou/PLM_PER/datasets/{TASK}'
    subset_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    result_dict = {}





BATCH_SIZE = 4
SEQ_LEN = 1024

DATA_SHUFFLE = True
FREEZE_TEACHER = True

USE_BF16 = True

# MoE 的参数，目前训练的 2.8B-0.8B 模型的参数就是如下
USE_MOE = True
MOE_INTERMEDIATE_SIZE = 1536
EXPERT_PER_TOKEN = 2
NUM_EXPERTS = 16
GATING_REFERENCE = "oracle" # oracle / switch


DEVICE = 7
CONFIG_DIR = "/home/jxzhou/PLM_PER/qwen/Qwen3-0.6B" # 就直接用 qwen3 的 dir 就可以
# CKPT_DIR = "/home/fdong/lowmem_qwen/checkpoints" # checkpoint dir
CKPT_DIR = "/home/jxzhou/PLM_PER/qwen/checkpoints" # checkpoint dir

TEST_CHECKPOINT = "ultrachat-oracle-0.500.pth" # test checkpoint
# TEST_CHECKPOINT = "pretrain-oracle-0.18000.pth" # test checkpoint


@torch.no_grad()
def prepare_qwenmodel():
    print("Construct student model. 1")

    config = AutoConfig.from_pretrained(CONFIG_DIR, trust_remote_code = True)
    config.moe_intermediate_size = MOE_INTERMEDIATE_SIZE
    config.num_experts_per_tok = EXPERT_PER_TOKEN
    config.num_experts = NUM_EXPERTS
    config.gating_reference = GATING_REFERENCE
    config.norm_topk_prob = True
    config.use_moe = USE_MOE

    print("Construct student model.2")

    model = MyQwen3ForCausalLM(config).to(DEVICE)
    print("Construct student model. 3")
    # print("Construct student model. 4")

    print(f'rank {DEVICE} student model ok, params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B/{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B') # 
    if os.path.exists(f"{CKPT_DIR}/{TEST_CHECKPOINT}"):
        model.load_state_dict(torch.load(f"{CKPT_DIR}/{TEST_CHECKPOINT}", weights_only = True, map_location="cpu"))
    else:
        print('no ckpts!')



    return model




@torch.no_grad()
def prepare_teacher2(replace_layers, weight_dirs = None):
    with open(f"{LLM_DIR}/device_map.json", 'r') as f:
        device_map = dict(json.load(f))
    t_model = AutoModelForCausalLM.from_pretrained(LLM_DIR, torch_dtype = T_DTYPE, local_files_only = True, trust_remote_code=True, device_map = device_map)
    t_model.eval()
    t_model.requires_grad_(False)
    print(f'teacher model loaded.')


    for replace_layer, weight_dir in zip(replace_layers, weight_dirs):
        if weight_dir is None:
            # 重新随机初始化 teacher 的 MoE
            for params in t_model.model.layers[replace_layer].mlp.parameters():
                nn.init.uniform_(params, 0, .01)
        else:
            # 替换为我蒸馏后的 MoE 参数
            s_model = MyDeepseekV2MoE(2048, 1408 * 2, 1408, 'silu', EXPERT_NUM, TOPK_EXPERT, .001)
            s_model.load_state_dict(torch.load(weight_dir, map_location='cpu', weights_only=True), strict=False)
            s_model = s_model.to(T_DTYPE).to(t_model.model.layers[replace_layer].input_layernorm.weight.device)
            t_model.model.layers[replace_layer].mlp = s_model
    
    return t_model



class MyIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
        **kwargs):

        outputs = (hidden_states,)
        expert_label = None

        if output_attentions:
            outputs += (None,)

        if use_cache:
            outputs += (None,)
        
        if expert_label is not None:
            outputs += (expert_label,)

        return outputs

@torch.no_grad()
def prepare_teacher(replace_layers, weight_dir = None):
    with open(f"{LLM_DIR}/device_map.json", 'r') as f:
        device_map = dict(json.load(f))
    t_model = AutoModelForCausalLM.from_pretrained(LLM_DIR, torch_dtype = T_DTYPE, local_files_only = True, trust_remote_code=True, device_map = device_map)
    t_model.eval()
    t_model.requires_grad_(False)
    print(f'teacher model loaded.')

    replace_layer = replace_layers[0]
    if weight_dir is None:
        # 重新随机初始化 teacher 的 MoE
        for params in t_model.model.layers[replace_layer].mlp.parameters():
            nn.init.uniform_(params, 0, .01)
    else:
        # 替换为我蒸馏后的 MoE 参数   
        s_model = MyDeepseekV2MoE(2048, 1408 * 2, 1408, 'silu', EXPERT_NUM, TOPK_EXPERT, .001)
        s_model.load_state_dict(torch.load(weight_dir, map_location='cpu', weights_only=True), strict=False)
        s_model = s_model.to(T_DTYPE).to(t_model.model.layers[replace_layer].input_layernorm.weight.device)
        t_model.model.layers[replace_layer].mlp = s_model
    
    for layer in replace_layers[1:]:
        t_model.model.layers[layer] = MyIdentity()
    return t_model






def extract_answer(text, key):
    pattern = rf"{re.escape(key)}(.*?)(?:\n|$)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()[:10]
    return None


def evaluate_prediction(prediction, gold_answers):
    """返回预测值与所有参考答案的最大EM和F1"""
    em_scores = []
    f1_scores = []

    for gold in gold_answers:
        em = compute_exact(gold, prediction)
        f1 = compute_f1(gold, prediction)
        em_scores.append(em)
        f1_scores.append(f1)

    return max(em_scores), max(f1_scores)
def extract_last_qa_block(text):
    # 匹配所有 Question/Answer 对
    pattern = r"(Question: .*?\nAnswer: .*?)(?=\nQuestion:|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    return matches[-1].strip() if matches else ""

def get_ngrams(tokens, n=2):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def rouge_n_f1(prediction, reference, n=2):
    """
    手写 ROUGE-n F1（默认 ROUGE-2）
    
    Args:
        prediction (str): 生成文本
        reference (str): 参考摘要
        n (int): n-gram 大小（默认为 2）

    Returns:
        f1 (float): ROUGE-n 的 F1 分数
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    pred_ngrams = Counter(get_ngrams(pred_tokens, n))
    ref_ngrams = Counter(get_ngrams(ref_tokens, n))

    overlap = sum((pred_ngrams & ref_ngrams).values())
    total_pred = sum(pred_ngrams.values())
    total_ref = sum(ref_ngrams.values())

    if overlap == 0:
        return 0.0

    precision = overlap / total_pred if total_pred > 0 else 0
    recall = overlap / total_ref if total_ref > 0 else 0
    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def load_xsum_from_bbc_summary(folder_path):
    data = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".summary"):
            continue
        with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        summary = ""
        document_lines = []
        mode = None

        for line in lines:
            line = line.strip()
            if line.startswith("[SN]"):
                if line == "[SN]FIRST-SENTENCE[SN]":
                    mode = "summary"
                elif line == "[SN]RESTBODY[SN]":
                    mode = "document"
                else:
                    mode = None
                continue

            if mode == "summary" and not summary:
                summary = line  # 只取一行摘要
            elif mode == "document":
                document_lines.append(line)

        document = " ".join(document_lines).strip()
        summary = summary.strip()

        if summary and document:
            data.append({
                "document": document,
                "summary": summary
            })

    return Dataset.from_list(data)


















def benchmarktest(test_data, model,tokenizer, sub):


    ab_token_strs = ["▁A", "▁B", "▁C", "▁D"]  # Qwen/Qwen2 通常是 sentencepiece，用▁表示 token 边界
    ab_token_ids = tokenizer.convert_tokens_to_ids(ab_token_strs)
    processor = LogitsProcessorList([
    ForceChoiceProcessor(ab_token_ids, step=0)  # 限制第一个token只能是A/B/C/D
    ])
    dataloader = DataLoader(test_data, batch_size=1, num_workers=4)
    # model, tokenizer, dataloader = accelerator.prepare(model, tokenizer, dataloader)


    # 评估指标累积
    total_em, total_f1, total, correct = 0.0, 0.0, 0, 0

    for batch in tqdm(dataloader):
        # input_ids = batch["input_ids"].cuda()
        # attention_mask = batch["attention_mask"].cuda()
        gold_answers = batch["answers"][0]  # MMLU/DROP/BBH/XSUM
        model_inputs = batch['model_inputs']
        # print(model_inputs)
        # gold_answers = batch["answers"] # 

        # print(gold_answers)
        # print('####')
        total_answers = batch['sentence']
        # print(batch['document'])

        judement = False

        with torch.no_grad():
            s = time.time()
            output_ids = model.generate(
                input_ids=model_inputs["input_ids"].squeeze(0).to(model.device),
                attention_mask=model_inputs["attention_mask"].squeeze(0).to(model.device),
                max_new_tokens=NEW_TOKENS,
                no_repeat_ngram_size=3,   # 防止重复
                do_sample=True,           # 启用采样生成更自然的回答
                top_k=10,                # Top-k 采样
                temperature=0.2,          # 控制生成多样性
            )
            t = time.time()

            print(f"{t-s}s")
            print(model_inputs["input_ids"].squeeze(0).shape[1])
            output_ids = output_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            # # print(output_ids[::-1])
            # # print(tokenizer.decode(872))
            # # parsing thinking content
    #         try:
    #             # rindex finding 151668 (</think>)
    #             index = len(output_ids) - output_ids[::-1].index(151667)

    #         except ValueError:
    #             index = 0
    #         # # print(tokenizer.encode('assistant'))
    #         thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    #         content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    #         prediction = content


    #         # print(thinking_content)
    #         # print("####")
            
    #         # print(prediction)
    #         # print("$$$$")

    #         # outputs = model(input_ids=input_ids)
    #         # logits = outputs.logits[:, -1, :]  # shape: [1, vocab_size]

    #         # ab_token_ids = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"])
    #         # mask = torch.full_like(logits, float("-inf"))
    #         # for idx in ab_token_ids:
    #         #     mask[0, idx] = logits[0, idx]

    #         # probs = torch.softmax(mask, dim=-1)
    #         # next_token_id = torch.argmax(probs, dim=-1).item()
    #         # prediction = tokenizer.decode([next_token_id])
    #         # output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=NEW_TOKENS,use_cache=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)[0] ##mmlu:
    #         # output_ids = model.generate(
    #         #     input_ids=input_ids,
    #         #     attention_mask=attention_mask,
    #         #     max_new_tokens=NEW_TOKENS,
    #         #     use_cache=False,
    #         #     pad_token_id=tokenizer.pad_token_id,
    #         #     eos_token_id=tokenizer.eos_token_id,
    #         #     logits_processor=processor,
    #         #     do_sample = False
    #         # )[0]
    #         # prediction = tokenizer.decode(output_ids[-NEW_TOKENS-PAD:], skip_special_tokens=True).strip()##mmlu -5:
    #         # prediction = tokenizer.decode(output_ids, skip_special_tokens=True).strip()##mmlu -5:
    # # ##########MMLU/BBH
    #     # print(total_answers)
    #     # print("$$$$$$$$")
    #     # print(prediction)
    #     # print('########')
    #     # if keyword in prediction:
    #         # answer_part = prediction.split(keyword, 1)[1].strip()
    #     # prediction = extract_last_qa_block(prediction) ###BBH
    #     # print(prediction)
    #     # answer_part = extract_answer(prediction, keyword)
    #     # print(answer_part)
    #     # answer_part = prediction
    #     # if answer_part:
    #     #     if gold_answers in answer_part:
    #     #         correct += 1
    #     #         judement = True
    #     #     elif gold_answers=='True' and '1' in answer_part:
    #     #         correct += 1
    #     #         judement = True
    #     #     elif gold_answers=='False' and '0' in answer_part:
    #     #         correct += 1
    #     #         judement = True
    #     # total+=1
    #     # print(f"Sample {total}: Answer Part: {prediction}, Correct:{correct/total}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")

    #     # if total%10 == 0:
    #     #     print(f"Sample {total}: Answer Part: {prediction}, Correct:{correct/total}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
    # # result_dict[sub] = correct/total
    # # print(f'sub-dataset: {sub}, accuracy: {correct/total}')
    # ###########XSUM
    # #     answer_part = prediction
    # #     rouge2=rouge_n_f1(answer_part, gold_answers,n=1)
    # #     total += 1

    # #     # if total%10 == 0:
    # #     #     print(f"Sample {total}: Answer Part: {prediction}, Rouge2:{rouge2}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
    # #     print(f"Sample {total}: Rouge2:{rouge2}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
    # #     if total == 100:
    # #         break
    # # print(f'sub-dataset: {sub}, accuracy: {rouge2/total}')
    # ####DROP
    #     # answer_part = prediction
    #     # if answer_part:
    #     #     if gold_answers in answer_part:
    #     #         correct += 1
    #     #         judement = True
    #     # total+=1
    #     # # print(f"Sample {total}: Answer Part: {prediction}, Correct:{correct/total}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
    # ####XSUM
    #     answer_part = extract_answer(prediction, '</think>')
    #     answer_part = prediction

    #     correct += rouge1_recall(answer_part,gold_answers)
    #     total+=1
    #     print(f"Sample {total}: Correct:{correct/total}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
    #     if total >=100:
    #         break
    
    ####TRIVIAQA
        # string_list = [item[0] for item in batch['answers']]      
        # gold_answers = string_list
        # answer_part = prediction
        # # if triviaqa_match(prediction, gold_answers):
        # #     correct += 1
        # # print(gold_answers)
        # # print(f"Sample {total}: Answer Part: {prediction}, Correct:{correct/total}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
        # f1 = metric_max_over_ground_truths(f1_score, answer_part, gold_answers)
        # correct += f1
        # total+=1
    #####MAG
        # answer_part = prediction
        # op_gold_answers = batch['answers'][0][0]+'.'
        # sub_gold_answers = batch['answers'][1][0]
        # # print(sub_gold_answers)
        # if answer_part:
        #     if op_gold_answers in answer_part or sub_gold_answers in answer_part:
        #         correct += 1
        #         judement = True
        # total+=1
        # print(f"Sample {total}: Answer Part: {prediction}, Correct:{correct/total}, LLM Answer: {answer_part}, Truth Answer: {batch['answers']}, Judgement: {judement}")
        
        
    #     if total%10 == 0:
    #         print(f"Sample {total}: Correct:{correct/total}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
    # print(f'task:{sub}, f1, accuracy:{correct/total}')
        # prediction = prediction.strip()

        # # 多参考答案时，取 max
        # em = metric_max_over_ground_truths(exact_match_score, prediction, gold_answers)
        # f1 = metric_max_over_ground_truths(f1_score, prediction, gold_answers)

        # 累加指标
        # total_em += em
        # total_f1 += f1
        # total += 1

        # print(f"Sample {total}: Rouge:{100.0 * total_f1 / total:.2f}, LLM Answer: {prediction}, Truth Answer: {gold_answers}, Judgement: {judement}")

    # print(f"Exact Match (EM): {100.0 * total_em / total:.2f}")
    # print(f"F1 Score: {100.0 * total_f1 / total:.2f}")



def benchmarktest_oracle(test_data, oracle,tokenizer, sub):
    dataloader = DataLoader(test_data, batch_size=1)

    # 评估指标累积
    total_em, total_f1, total, correct = 0.0, 0.0, 0, 0

    for batch in tqdm(dataloader):
        
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        gold_answers = batch["answers"][0]  # MMLU/BBH/XSUM
        # gold_answers = batch["answers"] 
        # total_answers = batch['sentence']
        judement = False
        # print(gold_answers)

        ###oracle
        # batch_size, seq_length = input_ids.shape[:2]
        # past_key_values_length = 0
        # position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=source.device,)
        # position_ids = position_ids.unsqueeze(0)

        with torch.no_grad():
            output_ids = oracle.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=NEW_TOKENS,use_cache=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)[0] ##mmlu:3
            prediction = tokenizer.decode(output_ids[-NEW_TOKENS-PAD:], skip_special_tokens=True).strip() 
    ##########MMLU/BBH
        # print(prediction)
        # if keyword in prediction:
        #     answer_part = prediction.split(keyword, 1)[1].strip()
        answer_part = extract_answer(prediction, keyword)
        if gold_answers in answer_part:
            correct += 1
            judement = True
        total+=1
        # print(f"Sample {total}: Answer Part: {prediction}, Correct:{correct/total}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")

        if total%10 == 0:
            print(f"Sample {total}: Answer Part: {prediction}, Correct:{correct/total}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
    result_dict[sub] = correct/total
    print(f'sub-dataset: {sub}, accuracy: {correct/total}')
    # ###########XSUM
    #     answer_part = prediction
    #     rouge2=rouge_n_f1(answer_part, gold_answers,n=1)
    #     total += 1
    #     # if total%10 == 0:
    #     #     print(f"Sample {total}: Answer Part: {prediction}, Rouge1:{rouge2/total}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
    #     print(f"Sample {total}: Rouge1:{rouge2}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
    #     if total == 100:
    #         break
    # print(f'sub-dataset: {sub}, accuracy: {rouge2/total}')

if __name__ == "__main__":

    # # dataset = load_xsum_from_bbc_summary("/home/jxzhou/PLM_PER/datasets/XSUM/bbc-summary-data")
    # # dataset.save_to_disk("/home/jxzhou/PLM_PER/datasets/XSUM/processed")
    # torch.cuda.set_device(1)
    # torch.set_default_dtype(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("/home/jxzhou/PLM_PER/qwen/Qwen3-0.6B", trust_remote_code=True)
    # # # # # # # # # # # # # ##Qwen1.5-MoE-A2.7B
    # # # # # # # # # # # # # ##DeepSeek-16B-2.8B
    # deepseek = AutoModelForCausalLM.from_pretrained('/home/jxzhou/PLM_PER/qwen/Qwen3-0.6B', device_map="auto", trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained('/home/jxzhou/PLM_PER/qwen/Qwen3-0.6B', device_map="cuda:7", trust_remote_code=True)
    

    # tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)
    # qwen = prepare_qwenmodel()

    # test_data = DATASET[TASK](f'/home/jxzhou/PLM_PER/datasets/{TASK}', tokenizer)
    test_data = DATASET[TASK](f'/home/jxzhou/PLM_PER/datasets', tokenizer)##triviaqa
    benchmarktest(test_data, qwen,tokenizer,TASK)
    # # sub = 'GPQA'
    # benchmarktest(test_data, deepseek,tokenizer,'XSUM')
    # tokenizer = AutoTokenizer.from_pretrained(LLM_DIR, trust_remote_code=True)

    # replace_layers 里的 layer index 从 1 开始，其中模型的第 1 层是一个 dense，第 2 - 27 层是 MoE。
    # replace_layers 是一个列表，因为之前训练的时候可能希望把第 2、3 层蒸馏成一个新层，因此 [2, 3] 就会把原有第 2 层变成我们蒸馏的结果，然后第 3 层直接删掉，这样相当于把两层合并成一层。
    # replace_layers 也可以只输入一个元素，比如 [2]，这时候就只把第 2 层换掉，其余不动。
    # weight_dir 是要加载的我们的参数，checkpoints 目录下 layer_xx_yy 就是把第 xx-yy 层一起蒸馏成一个新层，后缀有 oracle_gate 就是用我们的方法初始化，否则就是随机初始化。
    # checkpoints 下面的所有模型都是我们的 gating 机制，只是初始化有分别。
    # weight_dir 参数也可以为 None，为 None 时就直接把 replace_layers 里指定的层重新高斯初始化
    # 上面这些功能就可以让我们测试包括随机重置某层、换成我们的模型之类的下游任务效果了。
    
    # 把原模型的第 2、3 层换成我们训练的一层：
    # oracle = prepare_teacher(replace_layers = [1], weight_dir="/home/jxzhou/PLM_PER/qwen/checkpoints/layer_2_2_oracle_gate/layer-2-2-20000-new.pth")
    # 现在（看loss）好使的checkpint：
    # layer_2_3_oracle_gate/25000.pth
    # layer_2_3/25000.pth
    # layer_4_5/6000.pth~16000.pth 之间看起来都差不多
    # layer_6_7/4000.pth~6000.pth 看起来也差不多

    # 把原模型的第 2 层重新初始化：
    # for i in range(1,27):
    #     oracle = prepare_teacher(replace_layers = [i], weight_dir=None)

        # ####MMLU/BBH

    # oracle = prepare_teacher2(replace_layers = [12, 19, 26], weight_dirs=[
    #     "/home/jxzhou/PLM_PER/qwen/checkpoints/layer_13_13_oracle_gate/layer-13-13-5000.pth",
    #     "/home/jxzhou/PLM_PER/qwen/checkpoints/layer_20_20_oracle_gate/layer-20-20-2000.pth",
    #     "/home/jxzhou/PLM_PER/qwen/checkpoints/layer_27_27_oracle_gate/layer-27-27-14000.pth"
    #     ])
    
    # for sub in subset_dirs:

    #     print(f'testing {sub}...')
    #     test_data = DATASET[TASK](f'/home/jxzhou/PLM_PER/datasets/{TASK}/{sub}/test', tokenizer)
    #     # test_data = Benchmark_BBH_FewShot(f'/home/jxzhou/PLM_PER/datasets/{TASK}/{sub}/test', tokenizer)


        
    #     benchmarktest(test_data, qwen,tokenizer,sub)
    #     # benchmarktest_oracle(test_data, oracle,tokenizer,sub)


    #     print(f'tested {sub}...')
        

    #     # ####XSUM
    #     # print(f'testing XSUM...')
    #     # test_data = DATASET[TASK](f'/home/jxzhou/PLM_PER/datasets/{TASK}', tokenizer)
    #     # benchmarktest_oracle(test_data, oracle,tokenizer,'XSUM')
    #     # print(f'tested XSUM...')






    #     # del oracle          # 删除模型引用
    #     # torch.cuda.empty_cache()  # 清空 CUDA 缓存


    # # # ######MMLU/BBH
    # # if accelerator.is_main_process:
    # print("Per-subset Results:")
    # for k in sorted(result_dict.keys()):
    #     print(f"{k:20s} : {result_dict[k]:.4f}")

    # # 计算平均值
    # scores = list(result_dict.values())
    # avg = sum(scores) / len(scores) if scores else 0.0
    # print("\nAverage Accuracy:", round(avg, 4))


    # numbers = []
    # with open('/home/jxzhou/PLM_PER/qwen/scripts/0519-oracle-2-2-MMLU.log', 'r') as file:
    #     for line in file:
    #         # 使用正则匹配 `accuracy: 数字` 的模式
    #         match = re.search(r'accuracy:\s*([0-9.]+)', line)
    #         if match:
    #             numbers.append(float(match.group(1)))
    # # plt.figure(0)
    # # plt.plot([i for i in range(1,27)], numbers)
    # # plt.plot([i for i in range(1,27)], [0.28 for i in range(1,27)],label='teacher')
    # # plt.legend()
    # # plt.ylabel('accuracy')
    # # plt.xlabel('random layer index')
    # # plt.savefig('0516-random-layer-accuracy.png')
    # print(sum(numbers)/len(numbers))
