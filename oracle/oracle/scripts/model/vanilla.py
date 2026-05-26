import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .components import MultiheadAttention, GeneralFeedForward

def get_vanilla_model(rank, batch, vocab_size, layer_num, embed_dim, heads_num, hidden_dim, window_size, expert_num, moe_at, ckpt_dir):
    model = VanillaMoEGPT(vocab_size, .1, layer_num, embed_dim, heads_num, hidden_dim, window_size, .1, .1, expert_num, moe_at).to(rank)

    if batch > 0:
        weights = torch.load(f'{ckpt_dir}/step{batch}.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(weights)

    model.to(rank)
    return model



class VanillaRouter(nn.Module):
    def __init__(self, in_dim, expert_num) -> None:
        super().__init__()
        self.router = nn.Linear(in_dim, expert_num)
        self.expert_num = expert_num
        self.balance_loss = 0

    def forward(self, x):
        # x = [bs, embed_dim]
        router_logits = self.router(x) # [bs, expert_num]
        router_logits = router_logits.reshape(-1, self.expert_num) # [bs, expert_num]
        idx = router_logits.argmax(dim=-1, keepdim=True) # [bs]
        router_logits = torch.gather(router_logits, 1, idx) # [bs, 1]
        router_logits = F.softmax(router_logits, dim=-1)
        # mask = torch.zeros_like(router_logits)
        # for i in range(idx.size(0)):
        #     mask[i][idx[i]] = 1
        # percentage_1 = mask.mean(dim=0) # [expert_num], 每个 expert 被分到了百分之多少数据
        # percentage_2 = router_logits.mean(dim=0) # [expert_num], 每个 expert 被分到了百分之多少数据
        # self.balance_loss += percentage_1 @ percentage_2 * self.expert_num

        return router_logits, idx

    def load_balance_loss(self):
        ret = self.balance_loss
        self.balance_loss = 0
        return ret


class VanillaMOEDecoder(nn.Module):
    def __init__(self, embed_dim, heads_num, hidden_dim, window_size, attn_drop, ff_drop, expert_num) -> None:
        super().__init__()
        embed_dim = embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, heads_num, window_size, attn_drop)

        self.experts = nn.ModuleList([GeneralFeedForward(embed_dim, hidden_dim, embed_dim, ff_drop) for _ in range(expert_num)])
        self.router = VanillaRouter(embed_dim, expert_num)
        self.expert_num = expert_num


    def forward(self, x, output_attn_output = False, output_expert_label = False):
        output = {}
        x = self.ln1(x + self.attn(x))
        if output_attn_output:
            output['attn_output'] = x

        bs, seq_len = x.size(0), x.size(1)
        x = x.reshape(bs * seq_len, -1)
        final_output = x.new_zeros(x.shape)
        router_logits, indices = self.router(x) # [bs * seq_len], [bs * seq_len, expert_num]
        if output_expert_label:
            output['expert_label'] = indices

        idx_list = [torch.eq(indices, i).nonzero(as_tuple=True)[0] for i in range(len(self.experts))]
        expert_output = [expert(x[idx_list[i], :]) for i, expert in enumerate(self.experts)]
        for i, idx in enumerate(idx_list):
            final_output[idx, :] = expert_output[i]

        # router_logits = router_logits.max(dim=-1)[0] # [bs, seq_len]
        final_output = final_output * router_logits # [bs, seq_len, embed_dim]
        final_output = final_output.reshape(bs, seq_len, -1)

        output['layer_output'] = final_output
        return output


class VanillaMoEGPT(nn.Module):
    def __init__(self, vocab_size, emb_drop, layer_num, embed_dim, heads_num, hidden_dim, window_size, attn_drop, ff_drop, expert_num, moe_at) -> None:
        super().__init__()
        self.max_len = window_size
        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size-1)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, embed_dim))
        self.dropout = nn.Dropout(emb_drop)
        self.decoders = nn.ModuleList([VanillaMOEDecoder(embed_dim, heads_num, hidden_dim, window_size, attn_drop, ff_drop, expert_num) for _ in range(layer_num)])

        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x, output_layer_input = False, output_attn_output = False, output_expert_label = False):
        x = self.get_decoder_output(x, len(self.decoders) - 1, output_layer_input=output_layer_input, output_attn_output = output_attn_output, output_expert_label = output_expert_label)
        x['model_output'] = self.fc(self.ln(x['model_output']))

        return x

    def get_decoder_output(self, x, i, prev = None, output_layer_input = False, output_attn_output = False, output_expert_label = False):
        output = {}
        if output_layer_input:
            output['layer_input'] = []
        if output_attn_output:
            output['attn_output'] = []
        if output_expert_label:
            output['expert_label'] = []
        if prev is None:
            seq_len = x.size(1) # x = [bs, seq_len, vocab_size]
            tok_x = self.tok_emb(x) # tok_emb = [bs, seq_len, embed_dim]
            pos_emb = self.pos_emb[:, :seq_len, :]
            x = self.dropout(tok_x) + pos_emb
            for layer in range(i + 1):
                if output_layer_input:
                    output['layer_input'].append(x)
                output_l = self.decoders[layer](x, output_attn_output = output_attn_output, output_expert_label = output_expert_label)
                if output_attn_output:
                    output['attn_output'].append(output_l['attn_output'])
                if output_expert_label:
                    output['expert_label'].append(output_l['expert_label'])
                x = output_l['layer_output']
            output['model_output'] = x
            return output
        else:
            return self.decoders[i](prev)

