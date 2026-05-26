import torch
import base_models
import torch.optim as optim
from accelerate import Accelerator
from transformers import BertConfig, get_cosine_schedule_with_warmup
from Dataset_new import Test_Data
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random
writer = SummaryWriter('1113-log')


def get_old_loss_output(model, val_loader):
    old_losses = []
    old_outputs = []
    EMBS = []
    for i, batch in enumerate(val_loader): 
        # if i%100 == 0: 
        batch = {key: tensor.to('cuda') for key, tensor in batch.items()}      
        with torch.no_grad():
            loss, _,_,output = model(**batch)
            old_losses.append(loss)
            embeddings = model.bert.embeddings(batch['input_ids'])
            old_outputs.append(output)
            EMBS.append(embeddings)

    
    return old_outputs, old_losses, EMBS




def train(config, dataset):
    train_loader = dataset.train_loader2
    val_loader1 = dataset.val_loader
    val_loader2 = dataset.val_loader2
    num_epochs = 10
    lr = 1e-4
    kd = random.sample(range(1, 10000), 5)
    for d in kd:
        # model0 = base_models.BertWithMOE0(config)
        model = base_models.BertWithMOE(config)
        model.load_state_dict(torch.load('/home/jxzhou/PLM_PER/BERT-CL-main/train1.pt'))
        # model0.load_state_dict(torch.load('/home/jxzhou/PLM_PER/BERT-CL-main/train1.pt'))
        # model.load_state_dict(model0.state_dict())

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.00, betas=[0.9, 0.999], eps=1e-6)
        num_updates = num_epochs * len(train_loader)
        
        accelerator = Accelerator()

        model, train_loader, val_loader1, optimizer = accelerator.prepare(model, train_loader, val_loader1, optimizer)
        # d =74
        old_outputs, old_losses, EMBS = get_old_loss_output(model, val_loader1)

    

        for epoch in range(num_epochs):
            model.train()
            distances = []
            loss_difference = []

            for i, batch in enumerate(train_loader):
                # print(i)
                if i==d:
                    EMBS1 = model.bert.embeddings(batch['input_ids'])
                    had = EMBS1
                    

                    train_loss, _, routes, new_output = model(**batch)
                    for ji in range(3):
                        had = model.bert.encoders.layers[ji](had, batch['attention_mask'])
                    # new_output=had
                    # EMBS1 = model.bert.embeddings(batch['input_ids'])
                    
                    for oi in range(len(old_outputs)):
                        distances.append(torch.cdist(old_outputs[oi].view(old_outputs[oi].size(0),-1), new_output.view(new_output.size(0),-1), p=2).item())
                    # for oi in range(len(EMBS)):
                    #     distances.append(torch.cdist(EMBS[oi].view(EMBS[oi].size(0),-1), EMBS1.view(EMBS1.size(0),-1), p=2).item())

                    # print(routes)
                    # print(routes[1].shape)
                        
                    optimizer.zero_grad()
                    accelerator.backward(train_loss)
                    optimizer.step()

                    _, old_losses_after_opt, _= get_old_loss_output(model, val_loader1)
                    loss_difference = [a.item() - b.item() for a,b in zip(old_losses_after_opt,old_losses)]
                    distances = torch.tensor(distances)
                    loss_difference = torch.tensor(loss_difference)
                    break
            sorted_indices = torch.argsort(distances)

            sorted_distances = distances[sorted_indices]
            sorted_loss_difference = loss_difference[sorted_indices]
            
            x = sorted_distances.cpu().numpy()
            y = sorted_loss_difference.cpu().numpy()
            
        plt.figure(d)

        plt.plot(x, y)
        plt.title('compare-tawny-1113-EMBS')
        plt.xlabel('output_distances')
        plt.ylabel('loss_difference')
        plt.savefig('compare-tawny-1113-1e-3-epoch10-i%d'%d + '.png')
            
        plt.figure(10001)
        plt.plot(x, y)
        plt.title('compare-tawny-1113')
        plt.xlabel('output_distances')
        plt.ylabel('loss_difference')
    plt.savefig('compare-tawny-1113-1e-3-epoch10' + '.png')


if __name__ == "__main__":
    config = BertConfig.from_json_file('config/bert-1113.json')
    dataset = Test_Data(config)
    
    train(config, dataset)