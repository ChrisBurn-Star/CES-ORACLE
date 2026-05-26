import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import ACTIVATIONS



class MyAddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss



class MyDeepseekV2MLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_size, activation):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias=False)
        self.act_fn = ACTIVATIONS[activation]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MyMoEGate(nn.Module):
    def __init__(self, hidden_dim, expert_num, topk, aux_alpha):
        super().__init__()
        self.top_k = topk
        self.n_routed_experts = expert_num
        self.alpha = aux_alpha
        self.seq_aux = False

        self.gating_dim = hidden_dim
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bs, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
    
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bs, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bs, seq_len, -1)
                ce = torch.zeros(bs, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bs, seq_len * aux_topk, device=hidden_states.device),).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


# copied from DeepSeek V2
class MyDeepseekV2MoE(nn.Module):
    def __init__(self, hidden_dim, shared_expert_hidden_sim, routed_expert_hidden_dim, expert_activation, expert_num, topk, aux_alpha):
        super().__init__()
        self.topk = topk
        self.experts = nn.ModuleList([MyDeepseekV2MLP(hidden_dim, routed_expert_hidden_dim, expert_activation) for i in range(expert_num)])
        self.gate = MyMoEGate(hidden_dim, expert_num, topk, aux_alpha)
        self.shared_experts = MyDeepseekV2MLP(hidden_dim, shared_expert_hidden_sim, expert_activation)


    def forward(self, gating_reference, hidden_states = None):
        if hidden_states is None:
            hidden_states = gating_reference
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(gating_reference)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        hidden_states = hidden_states.repeat_interleave(self.topk, dim=0)
        y = torch.empty_like(hidden_states)
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.to(hidden_states.dtype).view(*orig_shape)
        
        with torch.no_grad():
            shared_expert_output = self.shared_experts(identity)
        y += shared_expert_output
        y = MyAddAuxiliaryLoss.apply(y, aux_loss) if aux_loss is not None else y
        return y, topk_idx, aux_loss

