import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def get_gpt(rank, epoch, batch, vocab_size, layer_num, embed_dim, heads_num, window_size, ckpt_dir):
    model = GPT(vocab_size, .1, layer_num, embed_dim, heads_num, window_size, .1, .1)
    
    if epoch > 0 or batch > 0:
        weights = torch.load(f'{ckpt_dir}/{epoch}_{batch}.pth', map_location='cpu', weights_only=True)
        # print(get_ckpt_path(epoch, batch, domain_count, embed_dim, 'none'))
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        model.load_state_dict(weights)
    
    model.to(rank)
    return model


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, heads_num, window_size, attn_drop) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.heads_num = heads_num
        self.window_size = window_size
        assert embed_dim % heads_num == 0, 'Embedding dimension must be divisible by number of heads.'

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(attn_drop)
        self.register_buffer('mask',
            torch.tril(torch.ones(1, 1, self.window_size, self.window_size), diagonal=0)
        )

    def forward(self, x):
        bs = x.size(0)
        seq_len = x.size(1)

        # x = [bs, seq_len, embed_dim]
        k = self.key(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        q = self.query(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        v = self.value(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        # k, q, v = [bs, heads_num, seq_len, embed_dim // heads_num]

        # [b, h, n, d] * [b, h, d, n] = [b, h, n, n]
        attn = (torch.matmul(q, k.transpose(-2, -1))) / math.sqrt(self.embed_dim // self.heads_num)
        mask = self.mask[:, :, :seq_len, :seq_len] #[1, 1, n, n]
        attn = attn.masked_fill(mask == 0, float('-inf'))  # 不能填 inf，不然第一行全是 inf 就出 nan 了

        # [b, h, n, n] 代表了每一个 token 对其他 token 的 attention
        # attn[b, 0, n] = q[b, 0, d] * k[b, d, n] * mask_fill
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # [b, h, n, n] * [b, h, n, d] = [b, h, n, d]     x[b, 0, d] = attn[b, 0, n] * v[b, n, d]
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(bs, seq_len, self.embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_drop) -> None:
        super().__init__()
        self.feed_fwd = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(ff_drop)
        )

    def forward(self, x):
        return self.feed_fwd(x)


class GeneralFeedForward(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, ff_drop) -> None:
        super().__init__()
        self.feed_fwd = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim),
            nn.Dropout(ff_drop)
        )

    def forward(self, x):
        return self.feed_fwd(x)


class Decoder(nn.Module):
    def __init__(self, embed_dim, heads_num, window_size, attn_drop, ff_drop) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, heads_num, window_size, attn_drop)
        self.feed_fwd = FeedForward(embed_dim, ff_drop)

    def forward(self, x):
        if isinstance(x, tuple):
            x, _ = x
        x = self.get_attn_output(x)
        x = self.get_ffn_output(x)

        return x
    
    def get_attn_output(self, x):
        if isinstance(x, tuple):
            x, _ = x
        x = self.ln1(self.attn(x) + x)
        return x
    
    def get_ffn_output(self, x):
        x = self.ln2(self.get_ffn_output_wo_ln(x) + x)
        return x
    
    def get_ffn_output_wo_ln(self, x):
        if isinstance(x, tuple):
            x, _ = x
        x = self.feed_fwd(x)
        return x
    
    def ffn_ln(self, x):
        return self.ln2(x)


class StaticMOEDecoder(nn.Module):
    def __init__(self, embed_dim, heads_num, window_size, attn_drop, ff_drop, expert_num, pretrained_module = None) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, heads_num, window_size, attn_drop)

        self.feed_fwd = nn.ModuleList([FeedForward(embed_dim, ff_drop) for _ in range(expert_num)])
        self.expert_num = expert_num

        if pretrained_module is not None:
            self.ln1.load_state_dict(pretrained_module.ln1.state_dict())
            self.ln2.load_state_dict(pretrained_module.ln2.state_dict())
            self.attn.load_state_dict(pretrained_module.attn.state_dict())
            for i in range(expert_num):
                if isinstance(pretrained_module.feed_fwd, nn.ModuleList):
                    self.feed_fwd[i].load_state_dict(pretrained_module.feed_fwd[i].state_dict())
                elif isinstance(pretrained_module.feed_fwd, FeedForward):
                    self.feed_fwd[i].load_state_dict(pretrained_module.feed_fwd.state_dict())

    def forward(self, x):
        x, domain = x
        x = x + self.attn(x)
        x = self.ln1(x)

        res = []
        for d, sent in zip(domain, x):
            res.append(self.feed_fwd[d](sent))
        x = x + torch.stack(res, dim=0)
        x = self.ln2(x)

        return x, domain


class GPT(nn.Module):
    def __init__(self, vocab_size, emb_drop, layer_num, embed_dim, heads_num, window_size, attn_drop, ff_drop) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size-1)
        self.pos_emb = nn.Parameter(torch.zeros(1, window_size, embed_dim))
        self.dropout = nn.Dropout(emb_drop)
        self.decoders = nn.Sequential(*[Decoder(embed_dim, heads_num, window_size, attn_drop, ff_drop) for _ in range(layer_num)])
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.get_decoder_output(x, len(self.decoders) - 1)
        x = self.decode(x)

        return x

    def get_decoder_output(self, x, i, prev = None):
        if prev is None:
            x = self.embed(x)
            for j in range(i + 1):
                x = self.decoders[j](x)
            return x
        else:
            return self.decoders[i](prev)

    def get_attn_output(self, x, layer):
        x = self.get_decoder_output(x, layer - 1)
        # self.embed(x)
        # for j in range(layer):
        #     x = self.decoders[j](x)
        x = self.decoders[layer].get_attn_output(x)
        return x

    def decode(self, x):
        x = self.fc(self.ln(x))
        return x

    def embed(self, x):
        seq_len = x.size(1)
        # x = [bs, seq_len, vocab_size]
        tok_x = self.tok_emb(x)
        # tok_emb = [bs, seq_len, embed_dim]
        pos_emb = self.pos_emb[:, :seq_len, :]
        x = self.dropout(tok_x) + pos_emb
        return x

