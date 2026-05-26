from utils import OpenwebTestData
from model import get_vanilla_model
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import gc
import time
import tiktoken

DEVICE = 3


def load_data(size):
    data_files = ['../../data/cut_legal/legal256.txt', '../../data/cut_med/med256_0.txt', '../../data/cut_openweb/openweb256_0.txt']
    data = []
    tokenizer = tiktoken.get_encoding('r50k_base')
    for file in data_files:
        with open(file, 'r') as f:
            for _ in range(size):
                text = f.readline()
                source = tokenizer.encode(text)[:256]
                source += [tokenizer.max_token_value] * (256 - len(source))
                source = torch.tensor(source).long()
                data.append(source)

    data = torch.stack(data)
    return data


def cal_jacobian(model, layer, module, data, normalized=False):
    torch.cuda.empty_cache()
    model.requires_grad_(False)
    active_module = None
    if module == 'attn':
        active_module = model.decoders[layer].attn.key.weight
    elif module == 'expert0':
        active_module = model.decoders[layer].experts[0].feed_fwd[2].weight
    elif module == 'expert1':
        active_module = model.decoders[layer].experts[1].feed_fwd[2].weight
    elif module == 'expert2':
        active_module = model.decoders[layer].experts[2].feed_fwd[2].weight
    elif module == 'expert3':
        active_module = model.decoders[layer].experts[3].feed_fwd[2].weight

    active_module.requires_grad_(True)
    jacobian = []

    model.zero_grad()
    output = model(data.to(DEVICE)) # [batch, seq_len, vocab_size]
    output = output['model_output']

    data_len = output.size(0)
    for i in range(data_len):
        output[i:i+1].mean().backward(retain_graph=True)
        grad = active_module.grad.detach().clone()
        if grad is not None:
            grad = grad.flatten()
            if normalized:
                grad = grad / torch.norm(grad)
            jacobian.append(grad)
            model.zero_grad()

    if len(jacobian) == 0:
        s = np.zeros(data_len)
    else:
        jacobian = torch.stack(jacobian)
        u, s, v = torch.svd(jacobian)
        s = s.detach().cpu().numpy()
        if len(s) < data_len:
            s = np.pad(s, (0, data_len - len(s)))
    # breakpoint()

    return s


def draw_singular_figure(module, sing_val):
    plt.title(f'{module} singular value')
    for i in range(sing_val.shape[0]):
        plt.plot(sing_val[i], 'b-', label=f'layer {i + 1}', alpha = (i + 1) / sing_val.shape[0])
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('singular value')
    plt.savefig(f'../figure_data/singular/{module}_singular.png')
    plt.close()


def main4(jacobian_normed, singular_normed):
    layers = range(0, 12, 1)
    # dataloader = DataLoader(OpenwebTestData(256), batch_size=10, shuffle=True)
    data = load_data(10)
    model = get_vanilla_model(DEVICE, 0,  50257, 12, 768, 12, 3072, 256, 4, [0,1,2,3,4,5,6,7,8,9,10,11] , f'../checkpoints/analyze_vanilla3')
    modules = ['attn', 'expert0', 'expert1', 'expert2', 'expert3']
    sing_val = {module : [[] for _ in layers] for module in modules}
    for test_batch in range(0, 50000 + 1, 500):
        print(f'test_batch {test_batch}')
        if test_batch > 0:
            model.load_state_dict(torch.load(f'../checkpoints/analyze_vanilla3/step{test_batch}.pth', weights_only=True, map_location = 'cpu'))
        for layer in layers:
            print(f'layer {layer}')
            for module in modules:
                s_val = cal_jacobian(model, layer, module, data, normalized=jacobian_normed)
                sing_val[module][layer].append(s_val)

    for module in modules:
        # module = 'ffn2'
        sing_val[module] = np.array(sing_val[module])
        print(sing_val[module].shape)
        np.save(f'../figures/singular/{module}_multidomain_sig_val_normed3.npy' if jacobian_normed 
                else f'../figures/singular/{module}_multidomain_sig_val_unnormed3.npy', sing_val[module])

    # breakpoint()
    # for module in modules:
    #     figure_name = f'{module}_'
    #     if jacobian_normed:
    #         figure_name += 'normed_jaco'
    #     else:
    #         figure_name += 'unnormed_jaco'
    #     if singular_normed:
    #         figure_name += '_normed_sing'
    #         draw_singular_figure(figure_name, sing_val[module][:,:,0] / sing_val[module].sum(axis = -1)) # [layer, step, 768]
    #     else:
    #         figure_name += '_unnormed_sing'
    #         draw_singular_figure(figure_name, sing_val[module][:,:,0]) # [layer, step, 768]



if __name__ == '__main__':
    main4(False, False)


