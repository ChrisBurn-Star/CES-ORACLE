import torch
from model import get_vanilla_model
import torch.nn as nn
import os
import torch.optim as optim

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


def train_vanilla(rank, start_step, lr, dataloader, vocab_size, layer_num, embed_dim, heads_num, hidden_dim, window_size, expert_num, moe_at, ckpt_dir, max_steps, save_steps, warmup_steps):
    # 会自动传入一个参数比如 rank 表示第几个进程
    print(f"Run training on rank {rank}.")

    print(f'rank {rank} data ok.')
    model = get_vanilla_model(rank, start_step, vocab_size, layer_num, embed_dim, heads_num, hidden_dim, window_size, expert_num, moe_at, ckpt_dir)
    model.train()

    loss_fn = nn.CrossEntropyLoss(ignore_index = vocab_size - 1).to(rank)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps)

    print(f'rank {rank} model ok. params: {sum(p.numel() for p in model.parameters())}')

    for step, (source, target, domain_label) in enumerate(dataloader, 1):
        source = source.to(rank)
        target = target.to(rank)
        domain_label = domain_label.to(rank)
        optimizer.zero_grad()

        output = model(source)
        output_logits = output['model_output']
        loss = loss_fn(output_logits.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_step = step + start_step
        print(f'step: {train_step}, loss: {loss.item()}')
        if train_step % save_steps == 0:
            torch.save(model.state_dict(), f'{ckpt_dir}/step{train_step}.pth')
        if train_step >= max_steps:
            break


def test_vanilla(rank, start_batch, dataloader, vocab_size, layer_num, embed_dim, heads_num, hidden_dim, window_size, expert_num, moe_at, ckpt_dir):
    # 会自动传入一个参数比如 rank 表示第几个进程
    print(f"Running test on rank {rank}.")

    model = get_vanilla_model(rank, start_batch, vocab_size, layer_num, embed_dim, heads_num, hidden_dim, window_size, expert_num, moe_at, ckpt_dir)
    model.eval()

    print(f'rank {rank} data ok.')

    metric_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=vocab_size-1).to(rank)
    perp = []
    with torch.no_grad():
        for batch, (source, target, _) in enumerate(dataloader, 1):
            source = source.to(rank)
            target = target.to(rank)

            output_dict = model(source, output_layer_input = True, output_attn_output=True, output_expert_label=True)
            output = output_dict['model_output']
            # breakpoint()
            for i in range(output.size(0)):
                loss = metric_fn(output[i].reshape([-1, vocab_size]), target[i].reshape([-1]))
                loss = loss.exp()
                perp.append(loss.item())

    print(f'test_batch: {start_batch}, perplexity: {sum(perp)/len(perp)}')

    return sum(perp)/len(perp)



