import os
import torch
import torch.nn as nn

from utils import DeepSeekDistillation, DomainData, DomainData_DCLM, DomainData_reviews
from models import MyQwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup, AutoModelForCausalLM
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, DistributedSampler
# from transformer_engine.pytorch import fp8_autocast

import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

LOCAL_BATCH_SIZE = 12
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

NUM_WORKERS = 2
DEVICES = [0] # [0,1,2,3,4,5,6,7]
CUTS = {0:22,1:23,2:24,3:25,4:26,5:27,6:28,7:29} ##31: 22-29
MASTER_ADDR = "127.0.0.1"
PORT = "32321"
CONFIG_DIR = "/home/jxzhou/PLM_PER/distill/model-1b-for-data"
# DATA_DIR = "../../data/washed_data_0520/train" # 2879 * 4096 = 11,792,384
DATA_DIR = "/home/jxzhou/datasets/dclm/filtered_output" # 2879 * 4096 = 11,792,384
# DATA_DIR = "../../data/washed_data_0520/reviews2" # 2879 * 4096 = 11,792,384

CHECK_MODEL = "/home/jxzhou/PLM_PER/distill/checkpoints/dclm/0.40000.pth"
# CHECK_MODEL = "/home/fdong/distill/checkpoints/reviews/0.18000.pth"

DATA_DOMAIN = "dclm"
Loss_threshold1 = 3.5
Loss_threshold2 = 1.5

CKPT_DIR = f"../checkpoints/qwen/MyPretrain-Qwen3-1B-{DATA_DOMAIN}-clean"


@torch.no_grad()
def prepare_model(local_rank, world_size, device):
    # model = AutoModelForCausalLM.from_pretrained(CONFIG_DIR, trust_remote_code = True)
    # print(f'rank {local_rank} CUDA {device} student model ok, params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B/{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B') # 
    # breakpoint()
    config = AutoConfig.from_pretrained(CONFIG_DIR, trust_remote_code = True)
    # config.moe_intermediate_size = MOE_INTERMEDIATE_SIZE
    # config.num_experts_per_tok = EXPERT_PER_TOKEN
    # config.num_experts = NUM_EXPERTS
    # config.gating_reference = GATING_REFERENCE
    # config.norm_topk_prob = True
    config.use_moe = USE_MOE

    model_class = MyQwen3ForCausalLM
    model = model_class(config).to(device)
    orig_model = model_class(config) # AutoModelForCausalLM.from_pretrained("../../Qwen-family/Qwen3-0.6B", torch_dtype="auto", device_map = "cpu")

    orig_model.load_state_dict(torch.load(CHECK_MODEL, weights_only = True, map_location="cpu"))
    model.load_state_dict(orig_model.state_dict())


    # model.load_state_dict(torch.load(f"../checkpoints/qwen/MyPretrain-Qwen3-0.6B/0.3000.pth", weights_only = True), strict = True)
    if local_rank == 0:
        print(config)

    if USE_MOE:
        config.use_moe = False
        orig_model = model_class(config) # AutoModelForCausalLM.from_pretrained("../../Qwen-family/Qwen3-0.6B", torch_dtype="auto", device_map = "cpu")
        orig_model.load_state_dict(torch.load(f"../checkpoints/qwen/Mytrain-Qwen3-0.6B/0.6000.pth", weights_only = True, map_location="cpu"))
        config.use_moe = True

        model.lm_head.load_state_dict(orig_model.lm_head.state_dict())
        model.model.embed_tokens.load_state_dict(orig_model.model.embed_tokens.state_dict())
        model.model.norm.load_state_dict(orig_model.model.norm.state_dict())
        model.model.rotary_emb.load_state_dict(orig_model.model.rotary_emb.state_dict())

        if True:
            model.lm_head.requires_grad_(False)
            model.model.embed_tokens.requires_grad_(False)
            model.model.rotary_emb.requires_grad_(False)

        for layer_idx, (s_layer, t_layer) in enumerate(zip(model.model.layers, orig_model.model.layers)):
            s_layer.self_attn.load_state_dict(t_layer.self_attn.state_dict())
            s_layer.input_layernorm.load_state_dict(t_layer.input_layernorm.state_dict())
            s_layer.post_attention_layernorm.load_state_dict(t_layer.post_attention_layernorm.state_dict())
            s_layer.mlp.shared_expert.load_state_dict(t_layer.mlp.state_dict())

            if GATING_REFERENCE == "oracle":
                gate_weight = torch.load(f"../router/myqwen_router/layer{layer_idx}/router_10.pth", map_location="cpu")
                s_layer.mlp.gate.weight.copy_(gate_weight["gate.weight"])
                s_layer.mlp.gate.requires_grad_(False)

            if FREEZE_TEACHER:
                s_layer.self_attn.requires_grad_(False)
                s_layer.mlp.shared_expert.requires_grad_(False)

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
    # dataset = DeepSeekDistillation(DATA_DIR, SEQ_LEN, tokenizer)
    # dataset = DomainData(DATA_DIR, DATA_DOMAIN, SEQ_LEN, tokenizer)
    dataset = DomainData_DCLM(DATA_DIR, CUTS[DEVICES[0]], SEQ_LEN, tokenizer)
    # dataset = DomainData_reviews(DATA_DIR, DATA_DOMAIN, SEQ_LEN, tokenizer)
    

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=DATA_SHUFFLE)
    dataloader = DataLoader(dataset, batch_size=LOCAL_BATCH_SIZE, num_workers=NUM_WORKERS, sampler=sampler)

    print(f"rank {local_rank} data ok. Data Length {len(dataset)}.")
    return dataloader


def prepare_loss_optimizer(s_model):
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=151643, reduction='mean')
    optimizer = torch.optim.AdamW([p for p in s_model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 2000, 50000)
    scaler = torch.amp.GradScaler("cuda")
    
    return token_loss_fn, optimizer, lr_scheduler, scaler

@torch.no_grad()
def forward_step_with_loss_check(local_rank, device, global_batch_idx, local_batch_idx, source, target, model, token_loss_fn):
    source, target = source.to(device), target.to(device)
    batch_size, seq_length = source.shape[:2]
    past_key_values_length = 0
    position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).to(device)

    s_output = model(source, position_ids=position_ids, output_hidden_states = False)
    s_logits = s_output.logits
    
    loss = token_loss_fn(s_logits.view(-1, s_logits.size(-1)), target.reshape(-1))
    loss = loss.view(batch_size,seq_length).mean(1)

    bad_indices = torch.nonzero(torch.logical_or(loss > Loss_threshold1, loss < Loss_threshold2)).squeeze().tolist()
    if type(bad_indices) == int:
        bad_indices = [bad_indices]
    
    if USE_MOE:
        aux_loss = s_output.loss
    else:
        aux_loss = 0
    
    

    if local_rank == 0:
        print(f"batch: {global_batch_idx}-{local_batch_idx}, bad_indices: {bad_indices}, loss: {loss.mean(0):.3f}, aux_loss: {aux_loss:.4f}", flush=True)

    if GATING_REFERENCE == "oracle":
        return loss

    return loss, bad_indices


def update_step(optimizer, scheduler):
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

@torch.no_grad()
def thread_main(local_rank, world_size):
    device = DEVICES[local_rank]
    print(f"running on device {local_rank} CUDA {device}")
    model = prepare_model(local_rank, world_size, device)
    dataloader = prepare_data(local_rank, world_size)
    token_loss_fn= nn.CrossEntropyLoss(ignore_index=151643, reduction='none')
    
    gradient_accumulation_steps_warmup = GLOBAL_BATCH_SIZE_WARM_UP // LOCAL_BATCH_SIZE // world_size
    gradient_accumulation_steps_train = GLOBAL_BATCH_SIZE_TRAIN // LOCAL_BATCH_SIZE // world_size
    
    for local_batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
        if local_batch_idx < BATCH_WARMUP_STEP * gradient_accumulation_steps_warmup:
            global_batch_idx = local_batch_idx // gradient_accumulation_steps_warmup
        else:
            global_batch_idx = BATCH_WARMUP_STEP + (local_batch_idx - BATCH_WARMUP_STEP) // gradient_accumulation_steps_train
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=USE_BF16):
            loss, bad_indices = forward_step_with_loss_check(local_rank, device, global_batch_idx, local_batch_idx, source, target, model, token_loss_fn)





def main():
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    world_size = len(DEVICES)
    if world_size == 1:
        thread_main(0, 1)
    else:
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = PORT


        print(f"world_size {world_size}")
        torch.multiprocessing.spawn(thread_main, args=(world_size,), nprocs=world_size, join=True)



if __name__ == "__main__":
    main()
    
