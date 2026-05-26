import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
# from utils import DeepSeekDistillation, DomainData, DomainData_DCLM, DomainData_reviews,DomainData_20B
from models import MyQwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup, AutoModelForCausalLM
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, DistributedSampler
from transformer_engine.pytorch import fp8_autocast
import random
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

LOCAL_BATCH_SIZE = 8
GLOBAL_BATCH_SIZE_WARM_UP = 128
GLOBAL_BATCH_SIZE_TRAIN = 256
BATCH_WARMUP_STEP = 2000
SAVE_INTERVAL = 1000
SEQ_LEN = 1024
LR = 8e-5

DATA_SHUFFLE = False
FREEZE_TEACHER = True

USE_FP8 = False
USE_BF16 = True

USE_MOE = False
MOE_INTERMEDIATE_SIZE = 1536
EXPERT_PER_TOKEN = 2
NUM_EXPERTS = 16
GATING_REFERENCE = "switch" # oracle / switch

NUM_WORKERS = 4
DEVICES = [0, 1, 2, 3, 4, 5, 6, 7] # [0,1,2,3,4,5,6,7]
MASTER_ADDR = "127.0.0.1"
PORT = "32321"
CONFIG_DIR = "../model-3m-for-data"
DATA_DIR = "../../data/washed_data_0520/train" # 2879 * 4096 = 11,792,384


DATA_DOMAIN = "MIXED-DATA-K-COSINE"
CKPT_DIR = f"../checkpoints/qwen/MyPretrain-Qwen3-1B-{DATA_DOMAIN}"




class DomainDataset(Dataset):
    def __init__(self, dataset_dir, domain_name, tokenizer, LENS=500000, max_seq_len=1024, padding=True):
        super().__init__()
        self.files = sorted([f for f in os.listdir(dataset_dir) if f.startswith(domain_name) and f.endswith(".txt")])
        self.file_paths = [os.path.join(dataset_dir, f) for f in self.files]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding

        self.samples = []
        for fp in self.file_paths:
            with open(fp, encoding="utf-8") as f:
                self.samples.extend(f.readlines())
        self.samples = self.samples[:LENS]
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx].strip()
        if self.padding:
            token_ids = self.tokenizer(
                text, padding="max_length",
                max_length=self.max_seq_len + 1,
                truncation=True, return_tensors="pt"
            ).input_ids
            real_len = len(self.tokenizer(text).input_ids)
        else:
            token_ids = self.tokenizer(text, return_tensors="pt").input_ids
            real_len = len(token_ids[0])
        return token_ids[0, :-1], token_ids[0, 1:], real_len


# -----------------------
# 衰减函数
# -----------------------
def cosine_decay(step, total_steps, rate=1.0):
    return 0.5 * (1 + np.cos(np.pi * step / (total_steps * rate)))

def linear_decay(step, total_steps, rate=1.0):
    return max(0.0, 1 - step / (total_steps * rate))

def exp_decay(step, total_steps, rate=5.0):
    return np.exp(-rate * step / total_steps)


# -----------------------
# 混合 Dataset (无 random)
# -----------------------
class MixedDomainDataset(Dataset):
    def __init__(self, datasetA, datasetB, total_steps, batch_size, schedule="cosine", rate=1.0):
        super().__init__()
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.total_steps = total_steps
        self.rate = rate

        if schedule == "cosine":
            self.schedule_fn = lambda s: cosine_decay(s, total_steps, rate)
        elif schedule == "linear":
            self.schedule_fn = lambda s: linear_decay(s, total_steps, rate)
        elif schedule == "exp":
            self.schedule_fn = lambda s: exp_decay(s, total_steps, rate)
        else:
            raise ValueError("Unknown schedule")

        # step & k
        self._step = 0
        self._k = self.schedule_fn(self._step)
        # self.k_history = []

        # 总长度 = A+B（不会丢数据）
        self.length = len(datasetA) + len(datasetB)

        self.batch_size = batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 计算当前步的 k
        if self._step != idx // self.batch_size:
            self._step = idx // self.batch_size
            self._k = self.schedule_fn(self._step)
            # self.k_history.append(self._k)

        # 计算A应该分配多少样本
        a_quota = int(self._k * self.length)

        if idx % self.batch_size < a_quota:  # 前一部分取自 A
            true_idx = idx % len(self.datasetA)
            return self.datasetA[true_idx]
        else:              # 后一部分取自 B
            true_idx = idx % len(self.datasetB)
            return self.datasetB[true_idx]

    def get_k(self):
        return self._k

    # def get_k_history(self):
    #     return self.k_history





@torch.no_grad()
def prepare_model(local_rank, world_size, device):

    config = AutoConfig.from_pretrained(CONFIG_DIR, trust_remote_code = True)


    model_class = MyQwen3ForCausalLM
    model = model_class(config).to(device)
    
    if local_rank == 0:
        print(config)

    if world_size == 1:
        pass
    else:
        dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
        model = DDP(model, device_ids=[device], find_unused_parameters=USE_MOE)
    print("Construct student model.")

    print(f'rank {local_rank} CUDA {device} student model ok, params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B/{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B') # 
    return model


def prepare_data(local_rank, world_size):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)
    legal_ds = DomainDataset("./data", "legal", tokenizer, max_seq_len=1024)
    openweb_ds = DomainDataset("./data", "openweb", tokenizer, max_seq_len=1024)

    dataset = MixedDomainDataset(legal_ds, openweb_ds, total_steps=100000, schedule="cosine", rate=1.0)
    

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=LOCAL_BATCH_SIZE, num_workers=NUM_WORKERS, sampler=sampler)

    print(f"rank {local_rank} data ok. Data Length {len(dataset)}.")
    return dataloader, dataset


def prepare_loss_optimizer(s_model):
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=151643, reduction='mean')
    optimizer = torch.optim.AdamW([p for p in s_model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 2000, 50000)
    scaler = torch.amp.GradScaler("cuda")
    
    return token_loss_fn, optimizer, lr_scheduler, scaler


def forward_step(local_rank, device, global_batch_idx, local_batch_idx, source, target, model, token_loss_fn):
    source, target = source.to(device), target.to(device)
    batch_size, seq_length = source.shape[:2]
    past_key_values_length = 0
    position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).to(device)

    s_output = model(source, position_ids=position_ids, output_hidden_states = False)
    s_logits = s_output.logits
    

    loss = token_loss_fn(s_logits.view(-1, s_logits.size(-1)), target.reshape(-1))

    if USE_MOE:
        aux_loss = s_output.loss
    else:
        aux_loss = 0

    if local_rank == 0:
        print(f"batch: {global_batch_idx}-{local_batch_idx}, loss: {loss:.3f}, aux_loss: {aux_loss:.4f}", flush=True)

    if GATING_REFERENCE == "oracle":
        return loss

    return loss + aux_loss


def update_step(optimizer, scheduler):
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def thread_main(local_rank, world_size):
    device = DEVICES[local_rank]
    print(f"running on device {local_rank} CUDA {device}")
    model = prepare_model(local_rank, world_size, device)
    dataloader, mixed_data = prepare_data(local_rank, world_size)
    token_loss_fn, optimizer, lr_scheduler, scaler = prepare_loss_optimizer(model)
    
    gradient_accumulation_steps_warmup = GLOBAL_BATCH_SIZE_WARM_UP // LOCAL_BATCH_SIZE // world_size
    gradient_accumulation_steps_train = GLOBAL_BATCH_SIZE_TRAIN // LOCAL_BATCH_SIZE // world_size
    K_HISTORY = []
    for epoch in range(1):
        for local_batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
            K_HISTORY.append(mixed_data.get_k())
            print(mixed_data.get_k())
            if local_batch_idx < BATCH_WARMUP_STEP * gradient_accumulation_steps_warmup:
                global_batch_idx = local_batch_idx // gradient_accumulation_steps_warmup
            else:
                global_batch_idx = BATCH_WARMUP_STEP + (local_batch_idx - BATCH_WARMUP_STEP) // gradient_accumulation_steps_train
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=USE_BF16):
                loss = forward_step(local_rank, device, global_batch_idx, local_batch_idx, source, target, model, token_loss_fn)

            if world_size == 1:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            
            # breakpoint()
            if (local_batch_idx < BATCH_WARMUP_STEP * gradient_accumulation_steps_warmup and local_batch_idx % gradient_accumulation_steps_warmup == 0) or \
                local_batch_idx % gradient_accumulation_steps_train == 0:
                update_step(optimizer, lr_scheduler)

                if local_rank == 0 and global_batch_idx % SAVE_INTERVAL == 0:
                    # torch.save(model.module.state_dict(), f"{CKPT_DIR}/{epoch}.{global_batch_idx}.pth")
                    torch.save(model.state_dict(), f"{CKPT_DIR}/{epoch}.{global_batch_idx}.pth")
                    
                    # torch.save(optimizer.state_dict(), f"{CKPT_DIR}/op_{epoch}.{global_batch_idx}.pth")

def main():
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    world_size = len(DEVICES)
    if world_size == 1:
        thread_main(0, 1)
    else:
        # os.environ['CUDA_VISIBLE_DEVICES']= str(DEVICES)[1:-1] # need to convert device list into str
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = PORT

        # world_size = torch.cuda.device_count()
        print(f"world_size {world_size}")
        torch.multiprocessing.spawn(thread_main, args=(world_size,), nprocs=world_size, join=True)


    # attn_output_tokens = torch.cat(attn_output_tokens, dim=0)
    # kmeans = cluster_tokens(attn_output_tokens, num_clusters=64)
    # cluster_counts = np.bincount(kmeans.labels_, minlength=64)
    # print(f'Cluster counts: {cluster_counts}')

    # cluster_centers = kmeans.cluster_centers_
    # analyse_cluster(torch.tensor(cluster_centers).to(3), torch.tensor(attn_output_tokens).to(3), kmeans.labels_)
    # breakpoint()


if __name__ == "__main__":
    main()
    
