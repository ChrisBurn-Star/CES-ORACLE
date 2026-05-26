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
from utils import compute_exact,compute_f1,Benchmark_BBH_FewShot_Qwen3,Benchmark_MMLU_FewShot_Qwen3,Benchmark_DROP, Benchmark_MMLU, Benchmark_GPQA,Benchmark_BBH,Benchmark_MMLU_FewShot,Benchmark_BBH_FewShot,Benchmark_XSUM_FewShot
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


os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'


# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,7'

BATCH_SIZE = 8
SEQ_LEN = 2048
EXPERT_NUM = 64
TOPK_EXPERT = 6
T_DTYPE = torch.bfloat16
S_DTYPE = torch.float32
GATING_REFERENCE = 'attn_output'
LLM_DIR = "/home/jxzhou/PLM_PER/qwen/DeepSeek-16B-2.8B"


TASK = 'BBH'
NEW_TOKENS = 5  ####MMLU 3; BBH 300 XSUM 30
PAD = 2 #####MMLU:5 BBH:5 XSUM:0
keyword = "Answer:"  ###MMLU
# keyword = "A:" ### BBH
# keyword = "Summary"  ###XSUM
DATASET = {'MMLU': Benchmark_MMLU_FewShot_Qwen3,'BBH':Benchmark_BBH_FewShot_Qwen3, 'XSUM/processed':Benchmark_XSUM_FewShot}

base_dir = f'/home/jxzhou/PLM_PER/datasets/{TASK}'
subset_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
result_dict = {}













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
        return match.group(1).strip()
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












def extract_last_qa_block(text):
    # 匹配所有 Question/Answer 对
    pattern = r"(Question: .*?\nAnswer: .*?)(?=\nQuestion:|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    return matches[-1].strip() if matches else ""





def benchmarktest(test_data, model,tokenizer, sub):


    ab_token_strs = ["▁A", "▁B", "▁C", "▁D"]  # Qwen/Qwen2 通常是 sentencepiece，用▁表示 token 边界
    ab_token_ids = tokenizer.convert_tokens_to_ids(ab_token_strs)
    processor = LogitsProcessorList([
    ForceChoiceProcessor(ab_token_ids, step=0)  # 限制第一个token只能是A/B/C/D
    ])
    dataloader = DataLoader(test_data, batch_size=1)

    # 评估指标累积
    total_em, total_f1, total, correct = 0.0, 0.0, 0, 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        gold_answers = batch["answers"][0]  # MMLU/DROP/BBH/XSUM
        model_inputs = batch['model_inputs']
        # print(model_inputs)
        # gold_answers = batch["answers"] # 

        # print(gold_answers)
        total_answers = batch['sentence']
        # print(batch['document'])

        judement = False

        with torch.no_grad():
            # output_ids = model.generate(
            #     input_ids=model_inputs["input_ids"].squeeze(0),
            #     attention_mask=model_inputs["attention_mask"].squeeze(0),
            #     max_new_tokens=NEW_TOKENS
            # )
            # output_ids = output_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            # # # print(output_ids[::-1])
            # # # print(tokenizer.decode(872))
            # # # parsing thinking content
            # try:
            #     # rindex finding 151668 (</think>)
            #     index = len(output_ids) - output_ids[::-1].index(151667)

            # except ValueError:
            #     index = 0
            # # # print(tokenizer.encode('assistant'))
            # # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            # content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            # prediction = content


            # print(thinking_content)
            # print("####")
            
            # print(prediction)
            # print("$$$$")

            # outputs = model(input_ids=input_ids)
            # logits = outputs.logits[:, -1, :]  # shape: [1, vocab_size]

            # ab_token_ids = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"])
            # mask = torch.full_like(logits, float("-inf"))
            # for idx in ab_token_ids:
            #     mask[0, idx] = logits[0, idx]

            # probs = torch.softmax(mask, dim=-1)
            # next_token_id = torch.argmax(probs, dim=-1).item()
            # prediction = tokenizer.decode([next_token_id])
            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=NEW_TOKENS,use_cache=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)[0] ##mmlu:
            # output_ids = output_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            # output_ids = model.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     max_new_tokens=NEW_TOKENS,
            #     use_cache=False,
            #     pad_token_id=tokenizer.pad_token_id,
            #     eos_token_id=tokenizer.eos_token_id,
            #     logits_processor=processor,
            #     do_sample = False
            # )[0]
            # prediction = tokenizer.decode(output_ids[-100:], skip_special_tokens=True).strip()##mmlu -5:
            prediction = tokenizer.decode(output_ids, skip_special_tokens=True).strip()##mmlu -5:

            # print(prediction)

    # ##########MMLU/BBH
        # print(total_answers)
        # print("$$$$$$$$")
        # print(prediction)
        # print('########')
        # if keyword in prediction:
        #     answer_part = prediction.split(keyword, 1)[1].strip()
        # answer_part = extract_answer(prediction, keyword)
        prediction = extract_last_qa_block(prediction) ###BBH
        # print(prediction)
        answer_part = extract_answer(prediction, keyword)

        # answer_part = prediction
        # print(answer_part)
        if answer_part:
            if gold_answers in answer_part:
                correct += 1
                judement = True
            elif gold_answers=='True' and '1' in answer_part:
                correct += 1
                judement = True
            elif gold_answers=='False' and '0' in answer_part:
                correct += 1
                judement = True

        total+=1
        # print(f"Sample {total}: Answer Part: {prediction}, Correct:{correct/total}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")

        if total%10 == 0:
            print(f"Sample {total}: Answer Part: {prediction}, Correct:{correct/total}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
    result_dict[sub] = correct/total
    print(f'sub-dataset: {sub}, accuracy: {correct/total}')
    ###########XSUM
    #     answer_part = prediction
    #     rouge2=rouge_n_f1(answer_part, gold_answers)
    #     total += 1
    #     # if total%10 == 0:
    #     #     print(f"Sample {total}: Answer Part: {prediction}, Rouge2:{rouge2}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
    #     print(f"Sample {total}: Rouge2:{rouge2}, LLM Answer: {answer_part}, Truth Answer: {gold_answers}, Judgement: {judement}")
    #     if total == 100:
    #         break
    # print(f'sub-dataset: {sub}, accuracy: {rouge2/total}')
    


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

    # # # dataset = load_xsum_from_bbc_summary("/home/jxzhou/PLM_PER/datasets/XSUM/bbc-summary-data")
    # # # dataset.save_to_disk("/home/jxzhou/PLM_PER/datasets/XSUM/processed")
    # # torch.cuda.set_device(1)
    # # torch.set_default_dtype(torch.bfloat16)
    # tokenizer = AutoTokenizer.from_pretrained("/home/jxzhou/PLM_PER/qwen/Qwen3-1.7B-Base", trust_remote_code=True)
    # # # # # # # # # # # # # # ##Qwen1.5-MoE-A2.7B
    # # # # # # # # # # # # # # ##DeepSeek-16B-2.8B
    # # deepseek = AutoModelForCausalLM.from_pretrained('/home/jxzhou/PLM_PER/qwen/Qwen3-0.6B', device_map="auto", trust_remote_code=True)
    # qwen = AutoModelForCausalLM.from_pretrained('/home/jxzhou/PLM_PER/qwen/Qwen3-1.7B-Base', device_map="auto", trust_remote_code=True)
    
    # # test_data = DATASET[TASK](f'/home/jxzhou/PLM_PER/datasets/{TASK}', tokenizer)
    # # # sub = 'GPQA'
    # # benchmarktest(test_data, deepseek,tokenizer,'XSUM')
    # # tokenizer = AutoTokenizer.from_pretrained(LLM_DIR, trust_remote_code=True)

    # # replace_layers 里的 layer index 从 1 开始，其中模型的第 1 层是一个 dense，第 2 - 27 层是 MoE。
    # # replace_layers 是一个列表，因为之前训练的时候可能希望把第 2、3 层蒸馏成一个新层，因此 [2, 3] 就会把原有第 2 层变成我们蒸馏的结果，然后第 3 层直接删掉，这样相当于把两层合并成一层。
    # # replace_layers 也可以只输入一个元素，比如 [2]，这时候就只把第 2 层换掉，其余不动。
    # # weight_dir 是要加载的我们的参数，checkpoints 目录下 layer_xx_yy 就是把第 xx-yy 层一起蒸馏成一个新层，后缀有 oracle_gate 就是用我们的方法初始化，否则就是随机初始化。
    # # checkpoints 下面的所有模型都是我们的 gating 机制，只是初始化有分别。
    # # weight_dir 参数也可以为 None，为 None 时就直接把 replace_layers 里指定的层重新高斯初始化
    # # 上面这些功能就可以让我们测试包括随机重置某层、换成我们的模型之类的下游任务效果了。
    
    # # 把原模型的第 2、3 层换成我们训练的一层：
    # # oracle = prepare_teacher(replace_layers = [1], weight_dir="/home/jxzhou/PLM_PER/qwen/checkpoints/layer_2_2_oracle_gate/layer-2-2-20000-new.pth")
    # # 现在（看loss）好使的checkpint：
    # # layer_2_3_oracle_gate/25000.pth
    # # layer_2_3/25000.pth
    # # layer_4_5/6000.pth~16000.pth 之间看起来都差不多
    # # layer_6_7/4000.pth~6000.pth 看起来也差不多

    # # 把原模型的第 2 层重新初始化：
    # # for i in range(1,27):
    # #     oracle = prepare_teacher(replace_layers = [i], weight_dir=None)

    #     # ####MMLU/BBH

    # # oracle = prepare_teacher2(replace_layers = [12, 19, 26], weight_dirs=[
    # #     "/home/jxzhou/PLM_PER/qwen/checkpoints/layer_13_13_oracle_gate/layer-13-13-5000.pth",
    # #     "/home/jxzhou/PLM_PER/qwen/checkpoints/layer_20_20_oracle_gate/layer-20-20-2000.pth",
    # #     "/home/jxzhou/PLM_PER/qwen/checkpoints/layer_27_27_oracle_gate/layer-27-27-14000.pth"
    # #     ])

    # for sub in subset_dirs:

    #     print(f'testing {sub}...')
    #     test_data = DATASET[TASK](f'/home/jxzhou/PLM_PER/datasets/{TASK}/{sub}/test', tokenizer)


        
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
    # print("Per-subset Results:")
    # for k in sorted(result_dict.keys()):
    #     print(f"{k:20s} : {result_dict[k]:.4f}")

    # # 计算平均值
    # scores = list(result_dict.values())
    # avg = sum(scores) / len(scores) if scores else 0.0
    # print("\nAverage Accuracy:", round(avg, 4))


    numbers = []
    c = 0
    with open('/home/jxzhou/PLM_PER/qwen/scripts/0605-oracle-mmlu.log', 'r') as file:
        for line in file:
            # 使用正则匹配 `accuracy: 数字` 的模式
            match = re.search(r'accuracy:\s*([0-9.]+)', line)
            if match:
                numbers.append(float(match.group(1)))
                c+=1
            if c==19:
                break
    # plt.figure(0)
    # plt.plot([i for i in range(1,27)], numbers)
    # plt.plot([i for i in range(1,27)], [0.28 for i in range(1,27)],label='teacher')
    # plt.legend()
    # plt.ylabel('accuracy')
    # plt.xlabel('random layer index')
    # plt.savefig('0516-random-layer-accuracy.png')
    print(sum(numbers)/len(numbers))
