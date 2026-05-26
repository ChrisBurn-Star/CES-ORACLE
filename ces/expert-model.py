import sys
from einops import rearrange, repeat
sys.path.append("/home/jxzhou/PLM_PER")
from sklearn.decomposition import TruncatedSVD
from moe.experts import *
from transformer.Transformer import *
import torch.nn as nn
import torch
from kmeans_pytorch import kmeans

class new_expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = Attention(config)
        self.ffn = FeedForwardNetwork(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        att_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)
        return output
    

class new_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.k = config.num_experts
        self.experts = nn.ModuleList([new_expert(config) for i in range(self.k)])
        
    
    def route(self, hidden_states, cluster_centers):
        sentence_states = torch.mean(hidden_states, dim=1)
        votes ={key:0 for key in range(self.k)}
        
        for sentence_state in sentence_states:
            index =  torch.argmin((cluster_centers-sentence_state)**2)
            votes[index] += 1

        expert_id = max(votes, key = votes.get)
        return expert_id

    def routing(self, hidden_states):
        sentence_states = torch.mean(hidden_states, dim=1)
        latent_states, s, v, latent_dim = self.PCA_sentence(sentence_states)
        cluster_ids_x, cluster_centers = kmeans(X=latent_states, num_clusters=self.config.num_experts, distance='euclidean')
                
        return cluster_ids_x
    
    def PCA_sentence(self, sentence_states):
        n = sentence_states.shape[0]
        mean_states = torch.mean(sentence_states, dim=0)
        sentence_states = sentence_states - mean_states
        cov = 1/n*torch.matmul(sentence_states.T, sentence_states)
        s, v = torch.eig(cov, eigenvectors=True)
        s, indices = torch.sort(s, descending=True)
        s = torch.norm(s)
        v = v[:, indices]
        threshold = 0.8
        information_rate = 0
        for i in range(len(s)):
            information_rate += s[i].item()
            if information_rate > threshold:
                break
        v_main = v[:i+1]
        latent_states = torch.matmul(sentence_states, v_main)

        return latent_states, s, v, i
    
    # def forward(self, hidden_states, attention_mask):
    #     output = hidden_states.new_zeros(hidden_states.shape)
    #     routing = self.routing(hidden_states)
    #     indexes_list = [torch.eq(routing, i).nonzero(as_tuple=True)[0] for i in range(self.config.num_experts)]
    #     expert_output = [self.experts[i](hidden_states[indexes_list[i], :, :]) for i in range(self.config.num_experts)]
    #     for i in range(self.config.num_experts):
    #         output[indexes_list[i], :, :] = expert_output[i]
    #     return output

    def forward(self, hidden_states, attention_mask, center):
        expert_id = self.route(hidden_states, center)
        expert_output = self.experts[expert_id](hidden_states, attention_mask)
        
        return expert_output


class new_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([new_layer(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            seed = 42
            torch.manual_seed(seed)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels, cluster_centers):
        hidden_states = self.embeddings(input_ids)
        for i in range(len(self.layers)):
            hidden_states = self.layers[i](hidden_states, attention_mask, cluster_centers[i])
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores