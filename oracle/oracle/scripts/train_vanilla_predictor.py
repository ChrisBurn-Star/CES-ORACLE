from transformers import get_linear_schedule_with_warmup
import torch
import os

from strategy.predictor import MSEPredictor, CEPredictor
from utils import Tokenized_data
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from utils import ExperimentConfig
from model import get_vanilla_model

PREDICTOR_DEVICE = 'cuda:0'
NUM_EXPERTS = 4
ACTIVATED_EXPERTS = 1
HIDDEN_SIZE = 768
LAYER_NUM = 12


def cal_topk_acc(pred_topk, router_top4, is_reverse = False):
    # pred_logits: [batch_size * seq_len, topk]
    # router_logits: [batch_size * seq_len, 4]
    accs = []
    for pred, route in zip(pred_topk, router_top4):
        pred = set(pred.cpu().numpy())
        route = set(route.cpu().numpy())
        if is_reverse: # 预测不需要的 expert
            all_experts = set(range(NUM_EXPERTS))
            not_needed = all_experts - route
            not_pred = pred - route
            acc = len(not_needed & not_pred) / len(not_needed)
        else: # 预测需要的 expert
            acc = len(pred & route) / ACTIVATED_EXPERTS
            accs.append(acc)
    return sum(accs) / len(accs)


def train(epoch, train_data, test_data, llm, pred_model, pred_input_pos, loss_func, optimizer, lr_scheduler, ckpt_dir):
    for batch, (text, _, _) in enumerate(train_data, 1):
        optimizer.zero_grad()
        text = text.to(PREDICTOR_DEVICE)
        
        with torch.no_grad():
            output = llm(text, output_layer_input = True, output_attn_output = True, output_expert_label=True) # 我改代码了，output_attn_outputs=True，返回每层的输入

        loss = 0
        for start_layer in range(LAYER_NUM):
            pred_logits = []
            pred_input = output[pred_input_pos][start_layer].reshape(-1, HIDDEN_SIZE).to(PREDICTOR_DEVICE)
            pred_logits.append(pred_model(pred_input, start_layer)) # [layers, batch_size * seq_len, num_experts]

            pred_logits = torch.cat(pred_logits, dim=0)

            router_logits = torch.stack([label for label in output['expert_label']], dim=0).to(PREDICTOR_DEVICE)
            loss += pred_model.__class__.cal_loss(loss_func, router_logits, pred_logits, expert_num = NUM_EXPERTS, activated_expert_num = ACTIVATED_EXPERTS, is_router_index = True)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        print(f"Epoch {epoch}, Batch {batch}, Loss {loss.item()}")
        if (batch) % 200 == 0:
            torch.save(pred_model.state_dict(), f"{ckpt_dir}/{epoch}.{batch}.pth")
            test(test_data, llm, pred_model, pred_input_pos, epoch, batch, ckpt_dir, topk = 1)
            test(test_data, llm, pred_model, pred_input_pos, epoch, batch, ckpt_dir, topk = 2)

    torch.save(pred_model.state_dict(), f"{ckpt_dir}/{epoch}.{batch}.pth")



def test(test_data, llm, pred_model, pred_input_pos, epoch, batch, ckpt_dir, topk = 4):
    layer_wise_test_acc = [
        [
            [] for _ in range(len(pred_model.predictors))
        ] for _ in range(LAYER_NUM)
    ] # 从第 i 层开始，预测第 j 层的结果
    
    pred_model.load_state_dict(torch.load(f"{ckpt_dir}/{epoch}.{batch}.pth", weights_only=True))
    for test_batch, (text, _, _) in enumerate(test_data, 1):
        text = text.to(PREDICTOR_DEVICE)
        with torch.no_grad():
            output = llm(text, output_layer_input = True, output_attn_output = True, output_expert_label=True)

            for start_layer in range(LAYER_NUM):
                pred_input = output[pred_input_pos][start_layer].reshape(-1, HIDDEN_SIZE).to(PREDICTOR_DEVICE)
                pred_logits = pred_model(pred_input, start_layer) # [layers, batch_size * seq_len, num_experts]

                # calculate accuracy for pred and router_logits
                for target_layer in range(LAYER_NUM):
                    layer_pred_index = pred_model.__class__.cal_pred_index(pred_logits[target_layer], topk, expert_num = NUM_EXPERTS) # [batch_size * seq_len, topk]
                    router_index = output['expert_label'][target_layer].to(PREDICTOR_DEVICE).reshape(-1, 1) # [batch_size * seq_len, 1]                    
                    # if start_layer == 6 and target_layer >= 6:
                    #     print(f'Data {test_batch}, Layer {start_layer}-to-{target_layer}, pred: {layer_pred_index[0].cpu().numpy()}, router: {router_index[0].cpu().numpy()}')

                    accuracy = cal_topk_acc(layer_pred_index, router_index)
                    layer_wise_test_acc[start_layer][target_layer].append(accuracy)

    for i in range(LAYER_NUM):
        for j in range(len(pred_model.predictors)):
            print(f"Epoch {epoch}, Batch {batch}, Layer-{i}-to-{j}-top{topk}-Accuracy {sum(layer_wise_test_acc[i][j]) / len(layer_wise_test_acc[i][j]):.4f}")
    


if __name__ == "__main__":
    exp_config = ExperimentConfig('train_vanilla', '../configs/train_vanilla.yml', force_duplicate = True)
    
    moe_model = get_vanilla_model(PREDICTOR_DEVICE, 10000, exp_config.vocab_size, exp_config.layer_num, exp_config.embed_dim, exp_config.heads_num, exp_config.hidden_dim, exp_config.window_size, exp_config.expert_num, exp_config.moe_at, '../checkpoints/train_vanilla')
    qwen_param_num=  sum(p.numel() for p in moe_model.parameters())
    print(f"MoE model has {qwen_param_num} parameters.")

    train_data = DataLoader(Tokenized_data(exp_config.window_size, False), batch_size=2)
    test_data = DataLoader(Tokenized_data(exp_config.window_size, True), batch_size=2)

    # section = 1
    lr = 5e-5
    predictor_class = CEPredictor

    pred_input = 'layer_input'
    ckpt_dir = f'../checkpoints/{predictor_class.__name__}_vanillaMoE_lr{lr}_{pred_input}'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    pred_model = predictor_class(layers=LAYER_NUM, input_dim=HIDDEN_SIZE, num_experts=NUM_EXPERTS).to(PREDICTOR_DEVICE)
    loss_func = predictor_class.get_loss_func()
    optimizer = Adam(pred_model.parameters(), lr=5e-5)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 500, 1e4)

    train(0, train_data, test_data, moe_model, pred_model, pred_input, loss_func, optimizer, lr_scheduler, ckpt_dir)
    # test(test_data[:50], tokenizer, qwen, model, sections, 0, 10000, ckpt_dir)
        


