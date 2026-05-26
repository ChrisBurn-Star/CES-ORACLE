import math
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = {
    'relu' : nn.ReLU(),
    'gelu' : nn.GELU(),
    'tanh' : nn.Tanh(),
    'silu' : nn.SiLU(),
    'sigmoid' : nn.Sigmoid(),
    'identity' : nn.Identity()
}


class ROtaryPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, rope_theta):
        super().__init__()
        self.max_seq_len_cached = max_seq_len
        self.original_max_seq_len = max_seq_len
        self.rope_theta = rope_theta

    @torch.no_grad()
    def forward(self, q, k, position_ids):
        # Ensure that q and k have the same shape
        assert q.shape == k.shape, "Q and K must have the same shape."
        
        # Get the dimensions
        batch_size, seq_len, n_heads, head_dim = q.size()
        
        # Create position indices
        theta = 1.0 / (self.rope_theta ** (2 * torch.arange(0, head_dim, 2, dtype=q.dtype, device=q.device) / head_dim))
        # Compute the theta for each position index
        
        # Calculate sinusoidal embeddings
        sinusoid_inp = torch.einsum('i,j->ij', position_ids, theta)
        sin_emb = torch.sin(sinusoid_inp)
        cos_emb = torch.cos(sinusoid_inp)

        # Reshape for broadcasting
        sin_emb = sin_emb.unsqueeze(1).repeat(1, n_heads, 1)
        cos_emb = cos_emb.unsqueeze(1).repeat(1, n_heads, 1)

        # Split heads dim and seq dim for applying rotation
        q_split = q.reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)
        k_split = k.reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)
        # Apply rotation
        q_rotated = torch.stack([q_split[..., 0] * cos_emb - q_split[..., 1] * sin_emb,
                                q_split[..., 0] * sin_emb + q_split[..., 1] * cos_emb], dim=-1).reshape_as(q)
        k_rotated = torch.stack([k_split[..., 0] * cos_emb - k_split[..., 1] * sin_emb,
                                k_split[..., 0] * sin_emb + k_split[..., 1] * cos_emb], dim=-1).reshape_as(k)
        
        return q_rotated, k_rotated


class MultiheadAttention(nn.Module):
    def __init__(self, layer_idx, embed_dim, head_dim, q_heads_num, kv_heads_num, max_seq_len, rope_theta, attn_drop) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.embed_dim = embed_dim
        self.q_heads_num = q_heads_num
        self.kv_heads_num = kv_heads_num
        self.kv_group_repeat = q_heads_num // kv_heads_num
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        
        self.query = nn.Linear(embed_dim, q_heads_num * head_dim, bias = True)
        self.key = nn.Linear(embed_dim, kv_heads_num * head_dim, bias = True)
        self.value = nn.Linear(embed_dim, kv_heads_num * head_dim, bias = True)
        self.o_proj = nn.Linear(q_heads_num * head_dim, embed_dim, bias = False)

        self.rope = ROtaryPositionEmbedding(max_seq_len, rope_theta)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(attn_drop)
        self.register_buffer('mask',
            torch.tril(torch.ones(1, 1, max_seq_len, max_seq_len), diagonal=0)
        )

    def prefill(self, x, output_attn_score = False):
        bs, self.seq_len, _ = x.shape

        # x = [bs, seq_len, embed_dim]
        q = self.query(x).view(bs, self.seq_len, self.q_heads_num, self.head_dim)
        k = self.key(x).view(bs, self.seq_len, self.kv_heads_num, self.head_dim)
        self.cached_v = self.value(x).view(bs, self.seq_len, self.kv_heads_num, self.head_dim).transpose(1, 2)
        # k, q, v = [bs, heads_num, seq_len, embed_dim // heads_num]

        position_ids = torch.arange(self.seq_len, dtype=x.dtype, device=q.device)
        q, k = self.rope(q, k, position_ids)

        q = q.transpose(1, 2)
        self.cached_k = k.transpose(1, 2)

        return self.cal_attention(bs, self.seq_len, q, output_attn_score)

    def decode(self, x, output_attn_score = False):
        bs, seq_len, _ = x.shape

        # x = [bs, seq_len, embed_dim]
        q = self.query(x).view(bs, seq_len, self.q_heads_num, self.head_dim)
        k = self.key(x).view(bs, seq_len, self.kv_heads_num, self.head_dim)
        v = self.value(x).view(bs, seq_len, self.kv_heads_num, self.head_dim).transpose(1, 2)
        # k, q, v = [bs, heads_num, seq_len, embed_dim // heads_num]

        position_ids = torch.tensor([self.seq_len], dtype=x.dtype, device=q.device)
        q, k = self.rope(q, k, position_ids)
        self.seq_len += 1

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        self.cached_k = torch.concat([self.cached_k, k], dim = 2)
        self.cached_v = torch.concat([self.cached_v, v], dim = 2)

        return self.cal_attention(bs, seq_len, q, output_attn_score)


    def cal_attention(self, bs, seq_len, q, output_attn_score):

        k = torch.repeat_interleave(self.cached_k, self.kv_group_repeat, dim = 1)
        v = torch.repeat_interleave(self.cached_v, self.kv_group_repeat, dim = 1)

        # [b, h, n, d] * [b, h, d, n] = [b, h, n, n]
        attn_score = (torch.matmul(q, k.transpose(-2, -1))) / math.sqrt(self.head_dim)
        mask = self.mask[:, :, :seq_len, :seq_len] #[1, 1, n, n]
        attn_score = attn_score.masked_fill(mask == 0, float('-inf'))  # 不能填 inf，不然第一行全是 inf 就出 nan 了

        # [b, h, n, n] 代表了每一个 token 对其他 token 的 attention
        # attn[b, 0, n] = q[b, 0, d] * k[b, d, n] * mask_fill
        attn_score = F.softmax(attn_score, dim=-1)

        # [b, h, n, n] * [b, h, n, d] = [b, h, n, d]     x[b, 0, d] = attn[b, 0, n] * v[b, n, d]
        x = torch.matmul(attn_score, v)
        x = x.transpose(1, 2).contiguous().view(bs, seq_len, self.embed_dim)
        x = self.o_proj(x)

        if output_attn_score:
            return x, attn_score
        return x


class TwoLayerMLP(nn.Module):
    def __init__(self, layer_idx, in_dim, hid_dim, out_dim, activation, drop_out) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer1 = nn.Linear(in_dim, hid_dim)
        self.activation = ACTIVATIONS[activation]
        self.layer2 = nn.Linear(hid_dim, out_dim)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.drop_out(x)
        
        return x


class GLU(nn.Module):
    def __init__(self, layer_idx, in_dim, hid_dim, out_dim, activation, drop_out) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.up_proj = nn.Linear(in_dim, hid_dim)
        self.gate = nn.Linear(in_dim, hid_dim)
        self.activation = ACTIVATIONS[activation]
        self.down_proj = nn.Linear(hid_dim, out_dim)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.activation(self.gate(x)) * self.up_proj(x)
        x = self.down_proj(x)
        x = self.drop_out(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, layer_idx, embed_dim, head_dim, q_heads_num, kv_heads_num, max_seq_len, rope_theta, attn_drop, mlp_hidden_dim, mlp_activation, mlp_drop) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(layer_idx, embed_dim, head_dim, q_heads_num, kv_heads_num, max_seq_len, rope_theta, attn_drop)
        self.mlp = TwoLayerMLP(layer_idx, embed_dim, mlp_hidden_dim, embed_dim, mlp_activation, mlp_drop)

    def forward(self, x, is_prefill = True, output_attention_score = False):
        if is_prefill:
            attn_output = self.attn.prefill(x, output_attention_score)
        else:
            attn_output = self.attn.decode(x, output_attention_score)
        if output_attention_score:
            attn_output, attention_score = attn_output
        x = self.ln1(attn_output + x)
        x = self.ln2(self.mlp(x) + x)

        if output_attention_score:
            return x, attention_score
        return x

