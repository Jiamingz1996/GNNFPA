import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import trange
from config import get_framework_parsers
from utils import get_geom_mask, get_few_mask, get_dataset, get_model, norm_adj

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()

def test(model, data):
    model.eval()
    logits, accs, losses, preds = model(data), [], [], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        loss = F.nll_loss(logits[mask], data.y[mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())
    return accs, preds, losses

def log_result(args, mean=10, std=0):
    dataset = args.dataset

    filename = f'./results/{dataset}_{args.model}.csv'
    path = os.path.join(os.path.dirname(__file__), 'results', f'{dataset}_{args.model}.csv')
    if os.path.exists(path):
        pass
    else:
        with open(f'{filename}', 'a+') as write_obj:
            write_obj.write('model ,' + 'train_rate ,' + 'val_rate ,' +
                'lr ,' + 'wd ,'  + 'hidden_channels ,' + 'dropout ,' +
                'alpha ,' + 'eta ,'  +'simple ,' +'acc\n')
    print(f'Saving results to {filename}')
    with open(f'{filename}', 'a+') as write_obj:
        write_obj.write(f'{args.model} ,' + f'{args.train_rate} ,' + f'{args.val_rate} ,' +
            f'{args.lr} ,' + f'{args.weight_decay} ,' + f'{args.hidden_channels} ,' + f'{args.dropout} ,'
            + f'{args.alpha} ,' + f'{args.eta} ,' + f'{args.simp} ,'f"""{mean:.2f} ±{std:.2f}""")

def runs(run, args, dataname, data, dataset):
    if args.dataset in ['cora', 'pubmed', 'citeseer']:
        pass
    else:
        if args.train_rate == 0.6:
            train_mask, test_mask, val_mask = get_geom_mask(dataname, run=run)
        else:
            train_rate = args.train_rate
            val_rate = args.val_rate
            num_per_class = int(round(train_rate * data.num_nodes / dataset.
                num_classes))
            num_val = int(round(val_rate * data.num_nodes))
            train_mask, test_mask, val_mask = get_few_mask(data, dataset.
                num_classes, num_per_class, num_val)
        data.train_mask = train_mask
        data.test_mask = test_mask
        data.val_mask = val_mask
    model = get_model(args, dataset, data, device)

    y_pred = F.one_hot(data.y).float()
    y_pred[~data.train_mask] = torch.zeros_like(y_pred[~data.train_mask])
    data.y_pred = y_pred

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(args.epochs):
        train(model, optimizer, data)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss,
            tmp_test_loss] = test(model, data)
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(val_loss_history[-(args.early_stopping +1):-1])
                if val_loss > tmp.mean().item():
                    break
    print(test_acc)
    result_all.append(test_acc)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='The Framework Parsers!')
    get_framework_parsers(parser)
    args = parser.parse_args()
    print(args)
    dataname = args.dataset
    dataset, data = get_dataset(dataname)
    data = data.to(device)
    print(type(data))
    result_all = []
    if args.dataset in ['cornell5', 'penn94', 'johnshopkins55']:
        data.y[data.y==-1] = 0 #Stands for Non-Homophily-Large-Scale datasets
    if args.dataset in['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'film']:
        data.adj = norm_adj(data)
    else:
        data.adj = None
    for run in trange(args.runs):
        runs(run, args, dataname, data, dataset)
    mean = np.mean(result_all) * 100
    std = np.std(result_all) * 100
    print(mean)
    log_result(args, mean, std)