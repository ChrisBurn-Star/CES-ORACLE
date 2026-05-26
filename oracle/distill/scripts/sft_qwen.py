import os
import torch
import torch.nn as nn

from utils import SFTDataset, OpenThoughtsSFTDataset, MMLUSFTDataset
from models import MyQwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup, AutoModelForCausalLM
from torch.utils.data import DataLoader, DistributedSampler

import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

LOCAL_BATCH_SIZE = 4
GLOBAL_BATCH_SIZE_WARM_UP = 128
GLOBAL_BATCH_SIZE_TRAIN = 256
BATCH_WARMUP_STEP = 100
SAVE_INTERVAL = 50
SEQ_LEN = 1024

DATA_SHUFFLE = True
FREEZE_TEACHER = True

USE_BF16 = True

USE_MOE = True
MOE_INTERMEDIATE_SIZE = 1536
EXPERT_PER_TOKEN = 2
NUM_EXPERTS = 16
GATING_REFERENCE = "oracle" # oracle / switch

NUM_WORKERS = 4
DEVICES = [0,1,2,3,4,5,6,7]
MASTER_ADDR = "127.0.0.1"
PORT = "32765"
RESUME_FROM = 0

CONFIG_DIR = "../../Qwen-family/Qwen3-0.6B" # 就直接用 qwen3 的 dir 就可以
PRETRAIN_CKPT_DIR = f"../checkpoints/qwen/MyPretrain-Qwen3-0.8B-2.8B-{GATING_REFERENCE}" # checkpoint dir
if GATING_REFERENCE == "oracle":
    PRETRAIN_CHECKPOINT = "0.18000.pth" # test checkpoint
elif GATING_REFERENCE == "switch":
    PRETRAIN_CHECKPOINT = "0.13000.pth" # test checkpoint

SFT_CKPT_DIR = f"../checkpoints/qwen/Ultrachat-ana-{GATING_REFERENCE}-from-{PRETRAIN_CHECKPOINT}/"
DATA_DIR = "../../data/sft/ultrachat"

@torch.no_grad()
def prepare_model(local_rank, world_size, device):
    config = AutoConfig.from_pretrained(CONFIG_DIR, trust_remote_code = True)
    config.moe_intermediate_size = MOE_INTERMEDIATE_SIZE
    config.num_experts_per_tok = EXPERT_PER_TOKEN
    config.num_experts = NUM_EXPERTS
    config.gating_reference = GATING_REFERENCE
    config.norm_topk_prob = True
    config.use_moe = USE_MOE

    model_class = MyQwen3ForCausalLM
    model = model_class(config).to(device)
    
    if local_rank == 0:
        print(config)

    if RESUME_FROM > 0:
        model.load_state_dict(torch.load(f"{SFT_CKPT_DIR}/0.{RESUME_FROM}.pth", weights_only = True, map_location = "cpu"))
    else:
        model.load_state_dict(torch.load(f"{PRETRAIN_CKPT_DIR}/{PRETRAIN_CHECKPOINT}", weights_only = True, map_location = "cpu"))
    
    if True:
        model.lm_head.requires_grad_(False)
        model.model.embed_tokens.requires_grad_(False)
        model.model.rotary_emb.requires_grad_(False)

    for layer_idx, s_layer in enumerate(model.model.layers):
        s_layer.mlp.gate.requires_grad_(False)
        s_layer.self_attn.requires_grad_(False)
        s_layer.mlp.shared_expert.requires_grad_(False)

    if world_size == 1:
        pass
    else:
        dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
        model = DDP(model, device_ids=[device], find_unused_parameters=USE_MOE)
    print("Construct student model.")

    print(f'rank {local_rank} student model ok, params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B/{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B') # 
    return model


def prepare_data(local_rank, world_size):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)
    dataset = SFTDataset(tokenizer, dataset_dir=DATA_DIR, max_seq_len=SEQ_LEN)

    print(f"Total global batches: {len(dataset) // GLOBAL_BATCH_SIZE_TRAIN}.")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=DATA_SHUFFLE)
    dataloader = DataLoader(dataset, batch_size=LOCAL_BATCH_SIZE, num_workers=NUM_WORKERS, sampler=sampler)
    return dataloader


def prepare_loss_optimizer(s_model):
    loss_fn = nn.CrossEntropyLoss(ignore_index = -100, reduction='mean')
    optimizer = torch.optim.AdamW([p for p in s_model.parameters() if p.requires_grad], lr=5e-5, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, 10000)
    scaler = torch.amp.GradScaler()

    if RESUME_FROM > 0:
        optimizer.load_state_dict(torch.load(f"{SFT_CKPT_DIR}/op_0.{RESUME_FROM}.pth", weights_only = True, map_location = "cpu"))
        lr_scheduler.load_state_dict(torch.load(f"{SFT_CKPT_DIR}/ls_0.{RESUME_FROM}.pth", weights_only = True, map_location = "cpu"))

    return loss_fn, optimizer, lr_scheduler, scaler


def forward_step(local_rank, device, global_batch_idx, local_batch_idx, input_ids, labels, model, loss_fn):
    input_ids, labels = input_ids.to(device), labels.to(device)
    batch_size, seq_length = input_ids.shape[:2]
    past_key_values_length = 0
    position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).to(device)

    output = model(input_ids, position_ids=position_ids)
    logits = output.logits
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.reshape(-1))

    if local_rank == 0:
        print(f"batch: {global_batch_idx}-{local_batch_idx}, loss: {loss:.5f}", flush=True)

    return loss


def update_step(optimizer, scheduler):
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def thread_main(local_rank, world_size):
    print(f"running on device {local_rank}")
    device = DEVICES[local_rank]
    dataloader = prepare_data(local_rank, world_size)
    model = prepare_model(local_rank, world_size, device)
    loss_fn, optimizer, lr_scheduler, scaler = prepare_loss_optimizer(model)
    
    step_batch_size = LOCAL_BATCH_SIZE * world_size
    gradient_accumulation_steps_warmup = GLOBAL_BATCH_SIZE_WARM_UP // step_batch_size
    gradient_accumulation_steps_train = GLOBAL_BATCH_SIZE_TRAIN // step_batch_size
    local_warmup_batch_num = BATCH_WARMUP_STEP * step_batch_size
    
    for epoch in range(100):
        for local_batch_idx, (input_ids, labels, prompt_lens, qa_lens) in enumerate(dataloader, 1):
            if local_batch_idx < local_warmup_batch_num:
                global_batch_idx = local_batch_idx // gradient_accumulation_steps_warmup
                minibatch_idx = local_batch_idx % gradient_accumulation_steps_warmup
            else:
                global_batch_idx = BATCH_WARMUP_STEP + (local_batch_idx - local_warmup_batch_num) // gradient_accumulation_steps_train
                minibatch_idx = (local_batch_idx - local_warmup_batch_num) % gradient_accumulation_steps_train
            
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=USE_BF16):
                loss = forward_step(local_rank, device, global_batch_idx, minibatch_idx, input_ids, labels, model, loss_fn)

            if world_size == 1:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            if (local_batch_idx < BATCH_WARMUP_STEP * gradient_accumulation_steps_warmup and local_batch_idx % gradient_accumulation_steps_warmup == 0) or \
                local_batch_idx % gradient_accumulation_steps_train == 0:
                update_step(optimizer, lr_scheduler)

                if local_rank == 0 and global_batch_idx % SAVE_INTERVAL == 0:
                    if world_size > 1:
                        torch.save(model.module.state_dict(), f"{SFT_CKPT_DIR}/{epoch}.{global_batch_idx}.pth")
                    else:
                        torch.save(model.state_dict(), f"{SFT_CKPT_DIR}/{epoch}.{global_batch_idx}.pth")
                    torch.save(optimizer.state_dict(), f"{SFT_CKPT_DIR}/op_{epoch}.{global_batch_idx}.pth")
                    torch.save(lr_scheduler.state_dict(), f"{SFT_CKPT_DIR}/ls_{epoch}.{global_batch_idx}.pth")
                    

def main():
    if not os.path.exists(SFT_CKPT_DIR):
        os.makedirs(SFT_CKPT_DIR)

    if len(DEVICES) == 1:
        thread_main(0, 1)
    else:
        # os.environ['CUDA_VISIBLE_DEVICES']= str(DEVICES)[1:-1] # need to convert device list into str
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = PORT

        world_size = torch.cuda.device_count()
        print(f"world_size {world_size}")
        torch.multiprocessing.spawn(thread_main, args=(world_size,), nprocs=world_size, join=True)



if __name__ == "__main__":
    main()
    
