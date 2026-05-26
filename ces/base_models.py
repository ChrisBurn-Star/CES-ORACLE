import torch.nn as nn
import torch
from transformer.Transformer import MoMoE_layer_0731_narrowneck,BertModel_showing,MoMoE_layer_0430_1_narrowneck,MoMoE_layer_0430_2_narrowneck,MoMoE_layer_0514_2_narrowneck,MoMoE_layer_0514_narrowneck,MoMoE_layer_0430_narrowneck,MoMoE_layer_0418_narrowneck,MoMoE_layer_0419_narrowneck,MoMoE_layer_0413_narrowneck,MoMoE_layer_0306_narrowneck_adl,MoMoE_layer_0411_narrowneck,MoMoE_layer_super_finegrained,MoMoE_layer_0315,MoMoE_layer_0313,MoMoE_layer_0306_narrowneck,MoMoE_layer_0229,MoMoE_layer_0226,MoMoE_layer_uniqueatt_commonffn,MoMoE_layer_0128,AngeL_rose_layer_0128,AngeL_rose_layer_salary1062,MoMoE_layer_salary1062,MoMoE_layer_0126,AngeL_rose_layer_0126,MoMoE_layer_prenorm,AngeL_rose_layer_prenorm,MoMoE_layer_speciallowrank,AngeL_rose_layer_speciallowrank,MoMoE_layer_commonandunique,MoMoE_layer,AngeL_rose_layer_tokens_cluster,AngeL_rose_layer_tokens,BertModel_toshow,AngeL_rose_layer,AngeLB_TransformerEncoders_Origin,AngeLB_TransformerEncoders_postnorm,AngeLB_PRE_DIS_TransformerEncoders,AngeL4_TransformerEncoders_postnorm,AngeL3_TransformerEncoders_postnorm,AngeLB_TransformerEncoders,AngeL3_TransformerEncoders,AngeL4_TransformerEncoders,BertModel, BertDecoderModel, BertLayerSaveModel, Embeddings, new_expert, new_layer, simple_layer, TransformerEncoder,vermilion_layer, rose_layer,BertModel_ELF,BertModel_ELF_Z, BertModel_ELF_B, BertModel_ELF_C,BertModel_postnorm,BertModel_prenorm,BertModelCombineResidual_postnorm,BertModelCombineResidual_prenorm,BertModel_prenorm
from transformers.models.bert.modeling_bert import BertOnlyMLMHead,BertPooler
# from Transformer_MOE import BertModelWithMOE

class BertForMLM(nn.Module):
    def __init__(self, config):
        super(BertForMLM, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output,O = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss,scores,O
    



class BertForMLM_showing(nn.Module):
    def __init__(self, config):
        super(BertForMLM_showing, self).__init__()
        self.config = config
        self.bert = BertModel_showing(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output,_ = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores

class BertForMLM_focalloss(nn.Module):
    def __init__(self, config):
        super(BertForMLM_focalloss, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.alpha = config.focalloss_alpha
        self.gamma = config.focalloss_gamma
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)
        old_loss = mlm_loss
        pt = torch.exp(-mlm_loss)
        mlm_loss = self.alpha * (1-pt)**self.gamma*mlm_loss
        return mlm_loss, scores,old_loss
    

class BertForMLM_prenorm(nn.Module):
    def __init__(self, config):
        super(BertForMLM_prenorm, self).__init__()
        self.config = config
        self.bert = BertModel_prenorm(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores


class BertForMLM_tawny(nn.Module):
    def __init__(self, config):
        super(BertForMLM_tawny, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores, output


class BertForMLM_ELF(nn.Module):
    def __init__(self, config):
        super(BertForMLM_ELF, self).__init__()
        self.config = config
        self.bert = BertModel_ELF(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output, ATTOUTS,ATTOS,FFNOS,OUTSA,FEB_ATT, FEB_FFN = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)
        #ATTOUTS: outputs of attention
        #ATTOS: outputs of attention+input
        #FFNOS: outputs of ffn
        #OUTSA: embeddings + output of layer(also the inputs, outputs of ffns+(output of attention+input))
        return mlm_loss, scores, ATTOUTS,ATTOS, FFNOS,OUTSA,FEB_ATT, FEB_FFN



####################1213 4exps###################


class BertForMLM_POST_O(nn.Module):
    def __init__(self, config):
        super(BertForMLM_POST_O, self).__init__()
        self.config = config
        self.bert = BertModel_postnorm(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output, ATTOUTS,ATTOS,FFNOS,OUTSA,FEB_ATT, FEB_FFN = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)
        #ATTOUTS: outputs of attention
        #ATTOS: outputs of attention+input
        #FFNOS: outputs of ffn
        #OUTSA: embeddings + output of layer(also the inputs, outputs of ffns+(output of attention+input))
        return mlm_loss, scores, ATTOUTS,ATTOS, FFNOS,OUTSA,FEB_ATT, FEB_FFN
class MoMoE_0430(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 2 or i == config.num_hidden_layers - 4:
                layers += [MoMoE_layer_0430_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        # self.router = nn.Linear(config.hidden_size, config.num_transformer)
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def forward(self, input_ids, attention_mask, labels):

        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0430_narrowneck):
                hidden_states = layer(hidden_states, attention_mask)
            else:
                raise ModuleNotFoundError
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores


class MoMoE_T_0514_1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 2 or i == config.num_hidden_layers - 4:
                layers += [MoMoE_layer_0430_1_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        # self.router = nn.Linear(config.hidden_size, config.num_transformer)
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def forward(self, input_ids, attention_mask, labels):

        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0430_1_narrowneck):
                hidden_states = layer(hidden_states, attention_mask)
            else:
                raise ModuleNotFoundError
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores



class MoMoE_T_0514_2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 2 or i == config.num_hidden_layers - 4:
                layers += [MoMoE_layer_0430_2_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        # self.router = nn.Linear(config.hidden_size, config.num_transformer)
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def forward(self, input_ids, attention_mask, labels):

        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0430_2_narrowneck):
                hidden_states = layer(hidden_states, attention_mask)
            else:
                raise ModuleNotFoundError
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores

class MoMoE_0413(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 2 or i == config.num_hidden_layers - 4:
                layers += [MoMoE_layer_0413_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        # self.router = nn.Linear(config.hidden_size, config.num_transformer)
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def forward(self, input_ids, attention_mask, labels,W_IDS):

        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0413_narrowneck):
                hidden_states = layer(hidden_states, attention_mask, W_IDS)
            else:
                raise ModuleNotFoundError
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores


class BertForMLM_POST_C(nn.Module):
    def __init__(self, config):
        super(BertForMLM_POST_C, self).__init__()
        self.config = config
        self.bert = BertModelCombineResidual_postnorm(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output, ATTOUTS,ATTOS,FFNOS,OUTSA,FEB_ATT, FEB_FFN = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)
        #ATTOUTS: outputs of attention
        #ATTOS: outputs of attention+input
        #FFNOS: outputs of ffn
        #OUTSA: embeddings + output of layer(also the inputs, outputs of ffns+(output of attention+input))
        return mlm_loss, scores, ATTOUTS,ATTOS, FFNOS,OUTSA,FEB_ATT, FEB_FFN

class BertForMLM_PRE_O(nn.Module):
    def __init__(self, config):
        super(BertForMLM_PRE_O, self).__init__()
        self.config = config
        self.bert = BertModel_prenorm(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output, ATTOUTS,ATTOS,FFNOS,OUTSA,FEB_ATT, FEB_FFN = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)
        #ATTOUTS: outputs of attention
        #ATTOS: outputs of attention+input
        #FFNOS: outputs of ffn
        #OUTSA: embeddings + output of layer(also the inputs, outputs of ffns+(output of attention+input))
        return mlm_loss, scores, ATTOUTS,ATTOS, FFNOS,OUTSA,FEB_ATT, FEB_FFN



class BertForMLM_PRE_C(nn.Module):
    def __init__(self, config):
        super(BertForMLM_PRE_C, self).__init__()
        self.config = config
        self.bert = BertModelCombineResidual_prenorm(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output, ATTOUTS,ATTOS,FFNOS,OUTSA,FEB_ATT, FEB_FFN = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)
        #ATTOUTS: outputs of attention
        #ATTOS: outputs of attention+input
        #FFNOS: outputs of ffn
        #OUTSA: embeddings + output of layer(also the inputs, outputs of ffns+(output of attention+input))
        return mlm_loss, scores, ATTOUTS,ATTOS, FFNOS,OUTSA,FEB_ATT, FEB_FFN












class BertForMLM_ELF_Z(nn.Module):
    def __init__(self, config):
        super(BertForMLM_ELF_Z, self).__init__()
        self.config = config
        self.bert = BertModel_ELF_Z(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output, ATTOUTS,ATTOS,FFNOS,OUTSA, FEB_ATT, FEB_FFN = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)
        #ATTOUTS: outputs of attention
        #ATTOS: outputs of attention+input
        #FFNOS: outputs of ffn
        #OUTSA: embeddings + output of layer(also the inputs, outputs of ffns+(output of attention+input))
        return mlm_loss, scores, ATTOUTS,ATTOS, FFNOS,OUTSA,FEB_ATT, FEB_FFN
class BertForMLM_ELF_B(nn.Module):
    def __init__(self, config):
        super(BertForMLM_ELF_B, self).__init__()
        self.config = config
        self.bert = BertModel_ELF_B(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def forward(self, input_ids, attention_mask, labels):
        output, ATTOUTS,ATTOS,FFNOS,OUTSA = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)
        #ATTOUTS: outputs of attention
        #ATTOS: outputs of attention+input
        #FFNOS: outputs of ffn
        #OUTSA: embeddings + output of layer(also the inputs, outputs of ffns+(output of attention+input))
        return mlm_loss, scores, ATTOUTS,ATTOS, FFNOS,OUTSA

class BertForMLM_ELF_C(nn.Module):
    def __init__(self, config):
        super(BertForMLM_ELF_C, self).__init__()
        self.config = config
        self.bert = BertModel_ELF_C(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output, ATTOUTS,ATTOS,FFNOS,OUTSA = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)
        #ATTOUTS: outputs of attention
        #ATTOS: outputs of attention+input
        #FFNOS: outputs of ffn
        #OUTSA: embeddings + output of layer(also the inputs, outputs of ffns+(output of attention+input))
        return mlm_loss, scores, ATTOUTS,ATTOS, FFNOS,OUTSA

class BertWithDecoders(nn.Module):
    def __init__(self, config):
        super(BertWithDecoders, self).__init__()
        self.config = config
        self.bert = BertDecoderModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.__init_weights)
    
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask)
        
        scores = self.head(outputs[0])
        outputs[0] = scores
        
        # replicated_labels = [labels for _ in range(len(outputs))]
        # losses = [self.criterion(output.view(-1, self.config.vocab_size), target.view(-1)) for output, target in zip(outputs, replicated_labels)]
        return outputs
    
    
class BertWithSavers(nn.Module):
    def __init__(self, config):
        super(BertWithSavers, self).__init__()
        self.config = config
        self.bert = BertLayerSaveModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.__init_weights)
    
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask, labels):
        output, layer_outputs, ffn_outputs = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores, layer_outputs, ffn_outputs
    


class BertWithSaversX(nn.Module):
    def __init__(self, config):
        super(BertWithSaversX, self).__init__()
        self.config = config
        self.bert = BertLayerSaveModel(config)
        self.head = BertOnlyMLMHead(config)
        self.pooler = BertPooler(config)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.__init_weights)
    
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask, labels):
        output, layer_outputs = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores, layer_outputs


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
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i])
            inputs[i][expert_id].append(hidden_states)
            # print(expert_id)
            hidden_states = self.layers[i](hidden_states, attention_mask, cluster_centers[i])
            outputs[i][expert_id].append(hidden_states)
            expert_ids.append(expert_id)
            
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids


class vermilion_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([vermilion_layer(config) for i in range(config.num_hidden_layers)])
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
        inputs = []
        outputs = []
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i])
            # print(expert_id)
            hidden_states = self.layers[i](hidden_states, attention_mask, cluster_centers[i])
            outputs.append(hidden_states)
            expert_ids.append(expert_id)
            
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids



class simple_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ahead_num = 3
        cluster_num = config.num_experts
        unique_dim = 160
        heads = 12
        self.layers1 = nn.ModuleList([TransformerEncoder(config) for i in range(ahead_num)])
        self.special_layer = simple_layer(self.config, cluster_num, unique_dim, 8)
        self.layers2 = nn.ModuleList([TransformerEncoder(config) for i in range(ahead_num+1, config.num_hidden_layers)])
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

    def forward(self, input_ids, attention_mask, labels, cluster_centers, pca):
        hidden_states = self.embeddings(input_ids)
        outputs = []
        for i in range(len(self.layers1)):
            hidden_states = self.layers1[i](hidden_states, attention_mask)
            outputs.append(hidden_states)
        hidden_states = self.special_layer(hidden_states, attention_mask, cluster_centers, pca)
        outputs.append(hidden_states)
        for j in range(len(self.layers2)):
            hidden_states = self.layers2[j](hidden_states, attention_mask)
            outputs.append(hidden_states)
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, outputs

class BertWithMOE0(nn.Module):
    def __init__(self, config):
        super(BertWithMOE0, self).__init__()
        self.config = config
        self.bert = BertModelWithMOE(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output, routes = self.bert(input_ids, attention_mask)
        raw_output = output
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores, routes
class BertWithMOE(nn.Module):
    def __init__(self, config):
        super(BertWithMOE, self).__init__()
        self.config = config
        self.bert = BertModelWithMOE(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output, routes = self.bert(input_ids, attention_mask)
        raw_output = output
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores, routes, raw_output
    


class rose_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([rose_layer(config) for i in range(config.num_hidden_layers)])
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

    # def forward(self, input_ids, attention_mask, labels, cluster_centers, hidden_states_for_router):
    #     hidden_states = self.embeddings(input_ids)
    #     inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
    #     outputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
    #     att_outs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
    #     ffn_outs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
    #     att_self = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
    #     expert_ids = []
    #     for i in range(len(self.layers)):
    #         expert_id = self.layers[i].route(hidden_states, cluster_centers[i], hidden_states_for_router[i])
    #         # print(expert_id)
    #         hidden_states,att_outputs,ffn_outputs,at_out_selfs = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hidden_states_for_router[i] )
    #         for d in range(hidden_states.shape[0]):
            
    #             att_outs[i][expert_id[d].item()].append(att_outputs[expert_id[d].item()])
    #             outputs[i][expert_id[d].item()].append(hidden_states[expert_id[d].item()])
    #             ffn_outs[i][expert_id[d].item()].append(ffn_outputs[expert_id[d].item()])
    #             att_self[i][expert_id[d].item()].append(at_out_selfs[expert_id[d].item()])
    #         # print(att_outs[i][0][1].shape)
    #         expert_ids.append(expert_id)
    #         for e in range(self.config.num_experts):
    #             if len(att_outs[i][e]):
    #                 att_outs[i][e] = torch.stack(att_outs[i][e])
    #                 outputs[i][e] = torch.stack(outputs[i][e])
    #                 ffn_outs[i][e] = torch.stack(ffn_outs[i][e])
    #                 att_self[i][e] = torch.stack(att_self[i][e])
    #     scores = self.head(hidden_states)
    #     mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

    #     return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self
    
    
    def forward(self, input_ids, attention_mask, labels, cluster_centers, hidden_states_for_router):
        hidden_states = self.embeddings(input_ids)
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i], hidden_states_for_router[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hidden_states_for_router[i] )
        
        
            att_outs.append(att_outputs)
            outputs.append(hidden_states)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            # print(att_outs[i][0][1].shape)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self

class cyan_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([rose_layer(config) for i in range(config.num_hidden_layers)])
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

    def forward(self, input_ids, attention_mask, labels, cluster_centers, hidden_states_for_router):
        hidden_states = self.embeddings(input_ids)
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i], hidden_states_for_router[i])
            # print(expert_id)
            hidden_states = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hidden_states_for_router[i] )

            expert_ids.append(expert_id)
            
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))
        return mlm_loss, scores, inputs, outputs, expert_ids
    


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = Flatten()
        self.fc1 = nn.Linear(784, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



###################AngeL MOE####################
class AngeL3_MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = AngeL3_TransformerEncoders(config)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels, routes):
        hidden_states = self.embeddings(input_ids)
        hidden_states,OUT_OF_ROUTER,ROUTES,router_labels = self.layers(hidden_states, attention_mask,routes)
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, OUT_OF_ROUTER,ROUTES,router_labels
class AngeL3_MoE_P(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = AngeL3_TransformerEncoders_postnorm(config)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels, routes):
        hidden_states = self.embeddings(input_ids)
        hidden_states,OUT_OF_ROUTER,ROUTES,router_labels = self.layers(hidden_states, attention_mask,routes)
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, OUT_OF_ROUTER,ROUTES,router_labels

class AngeL4_MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = AngeL4_TransformerEncoders(config)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels, routes):
        hidden_states = self.embeddings(input_ids)
        hidden_states,OUT_OF_ROUTER,ROUTES,router_labels = self.layers(hidden_states, attention_mask,routes)
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, OUT_OF_ROUTER,ROUTES,router_labels
    
class AngeL4_MoE_P(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = AngeL4_TransformerEncoders_postnorm(config)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels, routes):
        hidden_states = self.embeddings(input_ids)
        hidden_states,OUT_OF_ROUTER,ROUTES,router_labels = self.layers(hidden_states, attention_mask,routes)
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, OUT_OF_ROUTER,ROUTES,router_labels

class AngeLB_MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = AngeLB_TransformerEncoders(config)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels, routes):
        hidden_states = self.embeddings(input_ids)
        hidden_states,OUT_OF_ROUTER,ROUTES,router_labels,att_self = self.layers(hidden_states, attention_mask,routes)
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, OUT_OF_ROUTER,ROUTES,router_labels,att_self


class AngeLB_PRE_DIS_MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = AngeLB_PRE_DIS_TransformerEncoders(config)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels, routes):
        hidden_states = self.embeddings(input_ids)
        hidden_states,OUT_OF_ROUTER,ROUTES,router_labels = self.layers(hidden_states, attention_mask,routes)
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, OUT_OF_ROUTER,ROUTES,router_labels


class AngeLB_MoE_postnorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = AngeLB_TransformerEncoders_postnorm(config)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels, routes):
        hidden_states = self.embeddings(input_ids)
        hidden_states,OUT_OF_ROUTER,ROUTES,router_labels = self.layers(hidden_states, attention_mask,routes)
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, OUT_OF_ROUTER,ROUTES,router_labels



class AngeLB_MoE_Origin(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = AngeLB_TransformerEncoders_Origin(config)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels, routes):
        hidden_states = self.embeddings(input_ids)
        hidden_states,OUT_OF_ROUTER,ROUTES,router_labels = self.layers(hidden_states, attention_mask,routes)
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, OUT_OF_ROUTER,ROUTES,router_labels


############AngeL 5 MoE############

class AngeL_rose_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AngeL_rose_layer(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            seed = 45
            torch.manual_seed(seed)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels, cluster_centers):
        hidden_states = self.embeddings(input_ids)
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,output = self.layers[i](hidden_states, attention_mask, cluster_centers[i])
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(output)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self
    

class AngeL_rose_model_salary1062(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AngeL_rose_layer_salary1062(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        self.deval = 0
        
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            seed = 45
            torch.manual_seed(seed)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels):
        hidden_states = self.embeddings(input_ids)
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        for i in range(len(self.layers)):
            # expert_id = self.layers[i].route(hidden_states, cluster_centers[i])
            # print(expert_id)
            self.layers[i].deval = self.deval
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,output,expert_id = self.layers[i](hidden_states, attention_mask)
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(output)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self
    

class AngeL_rose_model_0126(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AngeL_rose_layer_0126(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            seed = 45
            torch.manual_seed(seed)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels, cluster_centers,hi_old):
        hidden_states = self.embeddings(input_ids)
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i],hi_old[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,output = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hi_old[i])
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(output)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self

class AngeL_rose_model_0128(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AngeL_rose_layer_0128(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        self.route = 0
        
    
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

    def forward(self, input_ids, attention_mask, labels):
        hidden_states = self.embeddings(input_ids)
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = [self.route for i in range(hidden_states.shape[0])]
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,output = self.layers[i](hidden_states, attention_mask, expert_id)
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(output)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self



class AngeL_rose_model_prenorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AngeL_rose_layer_prenorm(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            seed = 45
            torch.manual_seed(seed)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels, cluster_centers):
        hidden_states = self.embeddings(input_ids)
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,output = self.layers[i](hidden_states, attention_mask, cluster_centers[i])
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(output)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self
    




class AngeL_rose_model_sepciallowrank(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AngeL_rose_layer_speciallowrank(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            seed = 45
            torch.manual_seed(seed)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels, cluster_centers):
        hidden_states = self.embeddings(input_ids)
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,output = self.layers[i](hidden_states, attention_mask, cluster_centers[i])
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(output)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self


class AngeL_rose_model_tokens_cluster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AngeL_rose_layer_tokens_cluster(config) for i in range(config.num_hidden_layers)])
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

    def forward(self, input_ids, attention_mask, labels, cluster_centers, hidden_states_for_router):
        hidden_states = self.embeddings(input_ids)
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i], hidden_states_for_router[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hidden_states_for_router[i] )
            # print(hidden_states.shape)
            expert_ids.append(expert_id)
            outputs.append(hidden_states)
            att_outs.append(att_outputs)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self


class AngeL_rose_model_tokens(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AngeL_rose_layer_tokens(config) for i in range(config.num_hidden_layers)])
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

    def forward(self, input_ids, attention_mask, labels, cluster_centers, hidden_states_for_router):
        hidden_states = self.embeddings(input_ids)
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        att_outs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        ffn_outs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        att_self = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i], hidden_states_for_router[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hidden_states_for_router[i] )
            for d in range(hidden_states.shape[0]):
            
                att_outs[i][expert_id[d].item()].append(att_outputs[expert_id[d].item()])
                outputs[i][expert_id[d].item()].append(hidden_states[expert_id[d].item()])
                ffn_outs[i][expert_id[d].item()].append(ffn_outputs[expert_id[d].item()])
                att_self[i][expert_id[d].item()].append(at_out_selfs[expert_id[d].item()])
            # print(att_outs[i][0][1].shape)
            expert_ids.append(expert_id)
            for e in range(self.config.num_experts):
                if len(att_outs[i][e]):
                    att_outs[i][e] = torch.stack(att_outs[i][e])
                    outputs[i][e] = torch.stack(outputs[i][e])
                    ffn_outs[i][e] = torch.stack(ffn_outs[i][e])
                    att_self[i][e] = torch.stack(att_self[i][e])
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self
    


class BertForMLM_toshow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel_toshow(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def route(self, hidden_states, cluster_centers,hi_ou):
        sentence_states = torch.mean(hi_ou, dim=1)
        # expert_id = []
        # print(sentence_states.shape,cluster_centers.shape)
        distances = torch.cdist(sentence_states, cluster_centers)
        # print(distances.shape)
        nearest = torch.argmin(distances, dim=1)
        # print(nearest)
        expert_id = nearest
        # print(expert_id)
        return expert_id

    def forward(self, input_ids, attention_mask, label,cluster_centers):
        output,outputs,att_outs,ffn_outs,att_selfs,inputs = self.bert(input_ids, attention_mask)
        W_ids = self.route(output,cluster_centers,output)
        # scores = self.head(output)
        # mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)
        A= []
        B = []
        return A, B,outputs,att_outs,ffn_outs,att_selfs,inputs,W_ids
    


class MoMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer(config) for i in range(config.num_hidden_layers)])
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
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        inputs.append(hidden_states)
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,output = self.layers[i](hidden_states, attention_mask, cluster_centers[i])
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(output)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
            inputs.append(hidden_states)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self


class MoMoE_salary1062(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_salary1062(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        self.deval = 0
        
    
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

    def forward(self, input_ids, attention_mask, labels):
        hidden_states = self.embeddings(input_ids)
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        # inputs.append(hidden_states)
        for i in range(len(self.layers)):
            # expert_id = self.layers[i].route(hidden_states)
            # print(i)
            self.layers[i].deval = self.deval
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,output,expert_id = self.layers[i](hidden_states, attention_mask)
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(output)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
            # inputs.append(hidden_states)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self



class MoMoE_0126(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_0226(config) for i in range(config.num_hidden_layers)])
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

    def forward(self, input_ids, attention_mask, labels, cluster_centers,hi_ou,pre_vec):
        hidden_states = self.embeddings(input_ids)
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        cat_att = []
        common_atts = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i],hi_ou[i])
            # print(expert_id)
            

            hidden_states,att_outputs,ffn_outputs,at_out_selfs,input,CONCAT_ATTENTION_OUT,common_att = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hi_ou[i])
            cat_att.append(CONCAT_ATTENTION_OUT)
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)
            inputs.append(input)
            common_atts.append(common_att)
            att_outs.append(att_outputs)
            outputs.append(hidden_states)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self,cat_att,common_atts





class MoMoE_super_finegrained(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_super_finegrained(config) for i in range(config.num_hidden_layers)])
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

    def forward(self, input_ids, attention_mask, labels, cluster_centers,hi_ou,pre_vec):
        hidden_states = self.embeddings(input_ids)
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        cat_att = []
        common_atts = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i],hi_ou[i])
            # print(expert_id)
            

            hidden_states,att_outputs,ffn_outputs,at_out_selfs,input,CONCAT_ATTENTION_OUT,common_att = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hi_ou[i])
            cat_att.append(CONCAT_ATTENTION_OUT)
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)
            inputs.append(input)
            common_atts.append(common_att)
            att_outs.append(att_outputs)
            outputs.append(hidden_states)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self,cat_att,common_atts




class MoMoE_0313(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_0313(config) for i in range(config.num_hidden_layers)])
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

    def forward(self, input_ids, attention_mask, labels, cluster_centers,hi_ou,pre_vec):
        hidden_states = self.embeddings(input_ids)
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        cat_att = []

        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i],hi_ou[i])
            # print(expert_id)
            

            hidden_states,att_outputs,ffn_outputs,at_out_selfs,input,CONCAT_ATTENTION_OUT = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hi_ou[i])
            cat_att.append(CONCAT_ATTENTION_OUT)
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)
            inputs.append(input)

            att_outs.append(att_outputs)
            outputs.append(hidden_states)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self,cat_att


class MoMoE_0315(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_momoelayers = 4
        self.bert_layers = nn.ModuleList([TransformerEncoder(config) for i in range(config.num_hidden_layers-self.num_momoelayers)])
        self.layers = nn.ModuleList([MoMoE_layer_0315(config) for i in range(self.num_momoelayers)])
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

    def forward(self, input_ids, attention_mask, labels, cluster_centers,hi_ou,pre_vec):

        hidden_states = self.embeddings(input_ids)
        for j in range(self.config.num_hidden_layers-self.num_momoelayers):
            hidden_states = self.bert_layers[j](hidden_states,attention_mask)

        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        cat_att = []
        common_atts = []

        for i in range(self.num_momoelayers):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[self.config.num_hidden_layers-self.num_momoelayers+i],hi_ou[self.config.num_hidden_layers-self.num_momoelayers+i])
            # print(expert_id)
            

            hidden_states,att_outputs,ffn_outputs,at_out_selfs,input,CONCAT_ATTENTION_OUT = self.layers[i](hidden_states, attention_mask, cluster_centers[self.config.num_hidden_layers-self.num_momoelayers+i],hi_ou[self.config.num_hidden_layers-self.num_momoelayers+i])
            cat_att.append(CONCAT_ATTENTION_OUT)
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)
            inputs.append(input)

            att_outs.append(att_outputs)
            outputs.append(hidden_states)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self,cat_att,common_atts


class MoMoE_0229(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_0229(config) for i in range(config.num_hidden_layers)])
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

    def forward(self, input_ids, attention_mask, labels, cluster_centers,hi_ou,pro_vecs):
        hidden_states = self.embeddings(input_ids)
        inputs0 = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        cat_atts = []
        # inputs.append(hidden_states)
        for i in range(len(self.layers)):
            
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i],hi_ou[i],pro_vecs[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,cat_att = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hi_ou[i],pro_vecs[i])
            cat_atts.append(cat_att)
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)
            inputs0.append(inputs)
            att_outs.append(att_outputs)
            outputs.append(hidden_states)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
            # inputs.append(hidden_states)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs0, outputs, expert_ids,att_outs,ffn_outs,att_self,cat_atts



class MoMoE_0306_narrowneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_0306_narrowneck(config) for i in range(config.num_hidden_layers)])
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
    
    
    def forward(self, input_ids, attention_mask, labels, cluster_centers,W_IDS,pro_vec):
        hidden_states = self.embeddings(input_ids)
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        cat_att = []
        common_att = []

        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i],W_IDS)
            # print(expert_id)
            inputs.append(hidden_states)

            hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,CONCAT_ATTENTION_OUT,common_out = self.layers[i](hidden_states, attention_mask, cluster_centers[i],hi_ou[i])
            cat_att.append(CONCAT_ATTENTION_OUT)
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)
            common_att.append(common_out)
            att_outs.append(att_outputs)
            outputs.append(hidden_states)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self,cat_att,common_att


class MoMoE_0404(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # self.bert_layers = nn.ModuleList([TransformerEncoder(config) for _ in range(10)])
        self.layers = nn.ModuleList([MoMoE_layer_0306_narrowneck(config) for i in range(12)])
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
    
    
    def forward(self, input_ids, attention_mask, labels, W_IDS):
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        cat_att = []
        common_att = []
        hidden_states = self.embeddings(input_ids)

        


        for i in range(len(self.layers)):
            expert_id = W_IDS
            # print(expert_id)
            inputs.append(hidden_states)

            hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,CONCAT_ATTENTION_OUT,common_out = self.layers[i](hidden_states, attention_mask,W_IDS)
            cat_att.append(CONCAT_ATTENTION_OUT)
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)
            common_att.append(common_out)
            att_outs.append(att_outputs)
            outputs.append(hidden_states)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)


        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self,cat_att,common_att


class MoMoE_0411(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert_layers = nn.ModuleList([TransformerEncoder(config) for _ in range(10)])
        self.layers = nn.ModuleList([MoMoE_layer_0411_narrowneck(config) for i in range(2)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def forward(self, input_ids, attention_mask, labels, W_IDS):
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        cat_att = []
        common_att = []
        hidden_states = self.embeddings(input_ids)
        for i in range(9):
            hidden_states = self.bert_layers[i](hidden_states,attention_mask)

        
        expert_id = W_IDS
        # print(expert_id)
        inputs.append(hidden_states)

        hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,CONCAT_ATTENTION_OUT,common_out = self.layers[0](hidden_states, attention_mask,W_IDS)
        cat_att.append(CONCAT_ATTENTION_OUT)
        # print(hidden_states.shape)
        # print(len(at_out_selfs))
        # print(at_out_selfs[0].shape)
        # print(at_out_selfs[1].shape)
        common_att.append(common_out)
        att_outs.append(att_outputs)
        outputs.append(hidden_states)
        ffn_outs.append(ffn_outputs)
        att_self.append(at_out_selfs)
        expert_ids.append(expert_id)
        hidden_states = self.bert_layers[-1](hidden_states,attention_mask)

        expert_id = W_IDS
        # print(expert_id)
        inputs.append(hidden_states)

        hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,CONCAT_ATTENTION_OUT,common_out = self.layers[-1](hidden_states, attention_mask,W_IDS)
        cat_att.append(CONCAT_ATTENTION_OUT)
        # print(hidden_states.shape)
        # print(len(at_out_selfs))
        # print(at_out_selfs[0].shape)
        # print(at_out_selfs[1].shape)
        common_att.append(common_out)
        att_outs.append(att_outputs)
        outputs.append(hidden_states)
        ffn_outs.append(ffn_outputs)
        att_self.append(at_out_selfs)
        expert_ids.append(expert_id)

        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self,cat_att,common_att






# class MoMoE_0412(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         self.bert_layers = nn.ModuleList([TransformerEncoder(config) for _ in range(10)])
#         self.layers = nn.ModuleList([MoMoE_layer_0411_narrowneck(config) for i in range(2)])
#         self.embeddings = Embeddings(config)
#         self.head = BertOnlyMLMHead(config)
#         self.criterion = nn.CrossEntropyLoss()
#         self.router = nn.Linear(config.hidden_size, config.num_transformer)
#         self.router_trained = 1
        
#     def init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             seed = 42
#             torch.manual_seed(seed)
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
#                 with torch.no_grad():
#                     module.weight[module.padding_idx].fill_(0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
    
    
#     def forward(self, input_ids, attention_mask, labels, w_ids):
#         inputs = []
#         outputs = []
#         att_outs = []
#         ffn_outs = []
#         att_self = []
#         expert_ids = []
#         cat_att = []
#         common_att = []
#         hidden_states = self.embeddings(input_ids)
#         if self.router_trained:
#             _,W_IDS = torch.max(self.router(hidden_states),2)
#             print(W_IDS,W_IDS.shape)
#         else:
#             W_IDS = w_ids
        
#         for i in range(9):
#             hidden_states = self.bert_layers[i](hidden_states,attention_mask)

        
#         expert_id = W_IDS
#         # print(expert_id)
#         inputs.append(hidden_states)

#         hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,CONCAT_ATTENTION_OUT,common_out = self.layers[0](hidden_states, attention_mask,W_IDS)
#         cat_att.append(CONCAT_ATTENTION_OUT)
#         # print(hidden_states.shape)
#         # print(len(at_out_selfs))
#         # print(at_out_selfs[0].shape)
#         # print(at_out_selfs[1].shape)
#         common_att.append(common_out)
#         att_outs.append(att_outputs)
#         outputs.append(hidden_states)
#         ffn_outs.append(ffn_outputs)
#         att_self.append(at_out_selfs)
#         expert_ids.append(expert_id)
#         hidden_states = self.bert_layers[-1](hidden_states,attention_mask)

#         expert_id = W_IDS
#         # print(expert_id)
#         inputs.append(hidden_states)

#         hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,CONCAT_ATTENTION_OUT,common_out = self.layers[-1](hidden_states, attention_mask,W_IDS)
#         cat_att.append(CONCAT_ATTENTION_OUT)
#         # print(hidden_states.shape)
#         # print(len(at_out_selfs))
#         # print(at_out_selfs[0].shape)
#         # print(at_out_selfs[1].shape)
#         common_att.append(common_out)
#         att_outs.append(att_outputs)
#         outputs.append(hidden_states)
#         ffn_outs.append(ffn_outputs)
#         att_self.append(at_out_selfs)
#         expert_ids.append(expert_id)

#         scores = self.head(hidden_states)
#         mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

#         return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self,cat_att,common_att
class MoMoE_0413_onlyws(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 2 or i == config.num_hidden_layers - 4:
                layers += [MoMoE_layer_0418_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        # self.router = nn.Linear(config.hidden_size, config.num_transformer)
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def forward(self, input_ids, attention_mask, labels,W_IDS):

        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0418_narrowneck):
                hidden_states = layer(hidden_states, attention_mask, W_IDS)
            else:
                raise ModuleNotFoundError
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores


class MoMoE_0413_onlyffns(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 2 or i == config.num_hidden_layers - 4:
                layers += [MoMoE_layer_0418_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        # self.router = nn.Linear(config.hidden_size, config.num_transformer)
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def forward(self, input_ids, attention_mask, labels,W_IDS):

        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0418_narrowneck):
                hidden_states = layer(hidden_states, attention_mask, W_IDS)
            else:
                raise ModuleNotFoundError
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores




class MoMoE_0412(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 2 or i == config.num_hidden_layers - 4:
                layers += [MoMoE_layer_0411_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        # self.router = nn.Linear(config.hidden_size, config.num_transformer)
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def forward(self, input_ids, attention_mask, labels,W_IDS):

        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0411_narrowneck):
                hidden_states = layer(hidden_states, attention_mask, W_IDS)
            else:
                raise ModuleNotFoundError
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores




class MoMoE_0514(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 1 or i == config.num_hidden_layers - 2:
                layers += [MoMoE_layer_0514_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        # self.router = nn.Linear(config.hidden_size, config.num_transformer)
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def forward(self, input_ids, attention_mask, labels,W_IDS):

        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states,_ = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0514_narrowneck):
                hidden_states = layer(hidden_states, attention_mask, W_IDS)
            else:
                raise ModuleNotFoundError
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores


class MoMoE_0514_2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 1 or i == config.num_hidden_layers - 2:
                layers += [MoMoE_layer_0514_2_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        # self.router = nn.Linear(config.hidden_size, config.num_transformer)
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def forward(self, input_ids, attention_mask, labels,W_IDS):

        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states,_ = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0514_2_narrowneck):
                hidden_states = layer(hidden_states, attention_mask, W_IDS)
            else:
                raise ModuleNotFoundError
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores


class MoMoE_0731(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            if i ==config.num_hidden_layers - 1 or i == config.num_hidden_layers - 2:
                layers += [MoMoE_layer_0731_narrowneck(config)]
            else:
                layers += [TransformerEncoder(config)]
        self.layers = nn.ModuleList(layers)
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        # self.router = nn.Linear(config.hidden_size, config.num_transformer)
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    
    def forward(self, input_ids, attention_mask, labels,W_IDS):

        hidden_states = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states,_ = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoE_layer_0731_narrowneck):
                hidden_states = layer(hidden_states, attention_mask, W_IDS)
            else:
                raise ModuleNotFoundError
        
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores




class MoMoE_0404_adl(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert_layers = nn.ModuleList([TransformerEncoder(config) for _ in range(10)])
        self.layers = nn.ModuleList([MoMoE_layer_0306_narrowneck_adl(config) for i in range(2)])
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
    
    
    def forward(self, input_ids, attention_mask, labels, W_IDS):
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        cat_att = []
        common_att = []
        hidden_states = self.embeddings(input_ids)
        for i in range(9):
            hidden_states = self.bert_layers[i](hidden_states, attention_mask)


        expert_id = W_IDS

        inputs.append(hidden_states)

        hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,CONCAT_ATTENTION_OUT,common_out = self.layers[0](hidden_states, attention_mask,W_IDS)
        cat_att.append(CONCAT_ATTENTION_OUT)

        common_att.append(common_out)
        att_outs.append(att_outputs)
        outputs.append(hidden_states)
        ffn_outs.append(ffn_outputs)
        att_self.append(at_out_selfs)
        expert_ids.append(expert_id)

        hidden_states= self.bert_layers[-1](hidden_states, attention_mask)
        
        
        expert_id = W_IDS

        inputs.append(hidden_states)

        hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,CONCAT_ATTENTION_OUT,common_out = self.layers[-1](hidden_states, attention_mask,W_IDS)
        cat_att.append(CONCAT_ATTENTION_OUT)

        common_att.append(common_out)
        att_outs.append(att_outputs)
        outputs.append(hidden_states)
        ffn_outs.append(ffn_outputs)
        att_self.append(at_out_selfs)
        expert_ids.append(expert_id)


        # for i in range(len(self.layers)):
        #     expert_id = W_IDS
        #     # print(expert_id)
        #     inputs.append(hidden_states)

        #     hidden_states,att_outputs,ffn_outputs,at_out_selfs,inputs,CONCAT_ATTENTION_OUT,common_out = self.layers[i](hidden_states, attention_mask,W_IDS)
        #     cat_att.append(CONCAT_ATTENTION_OUT)
        #     # print(hidden_states.shape)
        #     # print(len(at_out_selfs))
        #     # print(at_out_selfs[0].shape)
        #     # print(at_out_selfs[1].shape)
        #     common_att.append(common_out)
        #     att_outs.append(att_outputs)
        #     outputs.append(hidden_states)
        #     ffn_outs.append(ffn_outputs)
        #     att_self.append(at_out_selfs)
        #     expert_ids.append(expert_id)


        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self,cat_att,common_att




class MoMoE_0128(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_0128(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        self.route = 0
        
    
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

    def forward(self, input_ids, attention_mask, labels):
        hidden_states = self.embeddings(input_ids)
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = [self.route for i in range(hidden_states.shape[0])]
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,output = self.layers[i](hidden_states, attention_mask, expert_id)
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(output)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self


class MoMoE_prenorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_prenorm(config) for i in range(config.num_hidden_layers)])
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
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,output = self.layers[i](hidden_states, attention_mask, cluster_centers[i])
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(output)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self


class MoMoE_speciallowrank(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_speciallowrank(config) for i in range(config.num_hidden_layers)])
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
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,output = self.layers[i](hidden_states, attention_mask, cluster_centers[i])
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(output)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self

class MoMoE_CAU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_commonandunique(config) for i in range(config.num_hidden_layers)])
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
        inputs = [[[]for j in range(self.config.num_experts)] for i in range(self.config.num_hidden_layers)]
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        for i in range(len(self.layers)):
            expert_id = self.layers[i].route(hidden_states, cluster_centers[i])
            # print(expert_id)
            hidden_states,att_outputs,ffn_outputs,at_out_selfs = self.layers[i](hidden_states, attention_mask, cluster_centers[i])
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(hidden_states)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self


class MoMoE_uniqueatt_commonffn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MoMoE_layer_uniqueatt_commonffn(config) for i in range(config.num_hidden_layers)])
        self.embeddings = Embeddings(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss()
        self.deval = 0
        
    
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

    def forward(self, input_ids, attention_mask, labels):
        hidden_states = self.embeddings(input_ids)
        inputs = []
        outputs = []
        att_outs = []
        ffn_outs = []
        att_self = []
        expert_ids = []
        # inputs.append(hidden_states)
        for i in range(len(self.layers)):
            # expert_id = self.layers[i].route(hidden_states)
            # print(i)
            self.layers[i].deval = self.deval
            hidden_states,att_outputs,ffn_outputs,at_out_selfs,expert_id = self.layers[i](hidden_states, attention_mask)
            
            # print(hidden_states.shape)
            # print(len(at_out_selfs))
            # print(at_out_selfs[0].shape)
            # print(at_out_selfs[1].shape)

            att_outs.append(att_outputs)
            outputs.append(hidden_states)
            ffn_outs.append(ffn_outputs)
            att_self.append(at_out_selfs)
            expert_ids.append(expert_id)
            # inputs.append(hidden_states)
        scores = self.head(hidden_states)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores, inputs, outputs, expert_ids,att_outs,ffn_outs,att_self
