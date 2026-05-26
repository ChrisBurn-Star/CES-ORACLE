import torch
from torch import nn


class MSEPredictor(nn.Module):
    def __init__(self, layers, input_dim, num_experts,):
        super(MSEPredictor, self).__init__()
        self.layers = layers
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.predictors = nn.ModuleList([nn.Linear(input_dim, num_experts, bias=True) for _ in range(layers)])

    def forward(self, x, start_layer = 0, end_layer = 24):
        output = []
        for i in range(start_layer, end_layer):
            output.append(self.predictors[i](x))
        
        output = torch.stack(output, dim=0)
        return output
    
    @classmethod
    def cal_loss(cls, mseloss, router_logits, pred_logits):
        # pred_logits: [layers, batch_size * seq_len, num_experts]
        # router_logits: [layers, batch_size * seq_len, num_experts]
        loss = mseloss(pred_logits, router_logits)
        return loss
    
    @classmethod
    def cal_pred_index(cls, pred_logits, topk = 4):
        # pred_logits: [batch_size * seq_len, num_experts]
        _, pred_topk = torch.topk(pred_logits, topk, dim=1)
        return pred_topk
    
    @classmethod
    def get_loss_func(cls, reduction = 'mean'):
        return nn.MSELoss(reduction=reduction)


class CEPredictor(nn.Module):
    def __init__(self, layers, input_dim, num_experts,):
        super(CEPredictor, self).__init__()
        self.layers = layers
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.predictors = nn.ModuleList()
        for i in range(layers):
            self.predictors.append(nn.ModuleList([nn.Linear(input_dim, num_experts * 2, bias=True) for _ in range(layers)]))


    def forward(self, x, start_layer):
        output = []
        for i in range(self.layers):
            output.append(self.predictors[start_layer][i](x))

        output = torch.stack(output, dim=0) # [layers, batch_size * seq_len, num_experts * 2]
        return output
    
    def predict(self, x, input_layer, target_layer, topk, expert_num):
        
        logits = self.predictors[input_layer][target_layer](x)
        
        output = CEPredictor.cal_pred_index(logits, topk, expert_num)
        return output

    @classmethod
    def cal_loss(cls, ce_loss, router_logits, pred_logits, expert_num = 60, activated_expert_num = 4, is_router_index = False):
        # pred_logits: [layers, batch_size * seq_len, num_experts * 2]
        # router_logits: [layers, batch_size * seq_len, num_experts]
        pred_logits = pred_logits.reshape([-1, expert_num * 2])
        pred_logits = pred_logits.reshape([-1, 2]) # [batch_size * seq_len * num_experts, 2]
        if is_router_index:
            router_index = router_logits
            router_index = router_index.reshape([-1, activated_expert_num]) # [batch_size * seq_len * num_experts, 1]
        else:
            router_logits = router_logits.reshape([-1, expert_num]) # [batch_size * seq_len, num_experts]
            _, router_index = torch.topk(router_logits, activated_expert_num, dim=-1) # [batch_size * seq_len, 4]
        
        # transfer router_index to one-hot label
        router_label = torch.zeros([pred_logits.shape[0]]).to(pred_logits.device).long()
        for i, index in enumerate(router_index):
            router_label.scatter_(0, i * expert_num + index, 1)
        loss = ce_loss(pred_logits, router_label)

        return loss

    @classmethod
    def cal_pred_index(cls, pred_logits, topk = 4, expert_num = 60):
        # pred_logits: [batch_size * seq_len, num_experts * 2]
        pred_logits = pred_logits.reshape([-1, 2]) # [batch_size * seq_len * num_experts, 2]
        pred_logits = torch.nn.functional.softmax(pred_logits, dim=-1)
        pred_logits = pred_logits[:,1]
        pred_logits = pred_logits.reshape([-1, expert_num]) # [batch_size * seq_len, num_experts]
        _, pred_index = torch.topk(pred_logits, topk, dim=-1)
        
        return pred_index # [batch_size * seq_len, topk]

    @classmethod
    def get_loss_func(cls, reduction = 'mean'):
        return nn.CrossEntropyLoss(reduction=reduction)

