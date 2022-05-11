import os
import os.path as osp
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.nn import GAE, VGAE, APPNP
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T

from VGNAE import *
from GeneralizedVGAE import *

def parse_boolean(value):
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGNAE')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--n_experiments', type=int, default=1)
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--normalize', type=parse_boolean, default=True)
parser.add_argument('--non_linear', type=parse_boolean, default=True)
parser.add_argument('--scaling_factor', type=float, default=1.8)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--result_file', type=str, default="/results/results_link_prediction_")
args = parser.parse_args()

file_path = os.getcwd() + args.result_file



if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='Planetoid', name=args.dataset, transform=NormalizeFeatures())
    data = dataset[0]
if args.dataset in ['cs', 'physics']:
    dataset = Coauthor(path, args.dataset, 'public')
    data = dataset[0]
    data = T.NormalizeFeatures()(data)
if args.dataset in ['computers', 'photo']:
    dataset = Amazon(path, args.dataset, 'public')
    data = dataset[0]
    data = T.NormalizeFeatures()(data)

out_channels = args.channels
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####### Define th
def train(model, optimizer, train_data, model_type):
    model.train()
    optimizer.zero_grad()
    z  = model.encode(train_data.x, train_data.pos_edge_label_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index) # + model.recon_loss(z, train_data.neg_edge_label_index)
    if model_type in ['VGNAE', 'Gen-VGNAE']:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss

def test(model, train_data, test_data):
    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.pos_edge_label_index)
    return model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)

def validation(model, train_data, val_data):
    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.pos_edge_label_index)
    return model.test(z, val_data.pos_edge_label_index, val_data.neg_edge_label_index)

results =[]
for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
#for training_rate in [0.2]:
    val_ratio = (1.0 - training_rate) / 3
    test_ratio = (1.0 - training_rate) / 3 * 2
    for exp in range(args.n_experiments):
        print(exp)
        transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                    is_undirected=True, split_labels=True)
        if args.model in ['Gen-VGNAE', 'Gen-GNAE']:
            alphas = alpha in np.arange(0,1.1, 0.1)
        else:
            alphas = [1.0, 1.5, 1.8, 2.0, 5.0, 10.]
        for alpha in alphas:
            train_data, val_data, test_data = transform(data)
            if args.model == 'Gen-GNAE':
                model = GAE(GCNEncoder(data.x.size()[1], out_channels, norm=True, alpha=alpha,
                beta=1.0, non_linearity=args.non_linear))
            elif args.model == 'Gen-VGNAE':
                model = VGAE(VariationalGCNEncoder(data.x.size()[1], out_channels, norm=True, alpha=alpha,
                                         beta=1.0, non_linearity=args.non_linear))
            elif args.model == 'GNAE':
                model = GAE(Encoder(data.x.size()[1], out_channels, train_data.pos_edge_label_index,
                        model='GNAE', scaling_factor=alpha)).to(dev)
            else:
                model = VGAE(Encoder(data.x.size()[1], out_channels, train_data.pos_edge_label_index,
                        model='VGNAE', scaling_factor=alpha)).to(dev)
            # move to GPU (if available)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            last_loss = 1e10
            patience = args.patience
            triggertimes = 0

            for epoch in range(1, args.epochs):
                loss = train(model, optimizer, train_data, args.model)
                loss = float(loss)
                with torch.no_grad():
                    test_pos, test_neg = train_data.pos_edge_label_index, train_data.neg_edge_label_index
                    auc, ap = test(model, train_data, test_data)
                    print('Epoch: {:03d}, LOSS: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))


                #### Add early stopping to prevent overfitting
                out = validation(model, train_data, val_data)
                current_loss = out[1]
                if current_loss >= last_loss:
                    trigger_times += 1
                    #print('Trigger Times:', trigger_times)
                    if trigger_times >= patience:
                        #print('Early stopping!\nStart to test process.')
                        break
                else:
                    #print('trigger times: 0')
                    trigger_times = 0
                last_loss = current_loss
            results += [[exp, args.model, args.dataset, args.non_linear, args.normalize, args.lr, args.channels,
                                  training_rate, val_ratio, test_ratio, alpha, auc, ap, epoch]]
            res1 = pd.DataFrame(results, columns=['exp', 'model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                                  'train_rate', 'val_ratio',
                                                  'test_ratio', 'alpha', 'auc', 'ap', 'epoch'])
            res1.to_csv(file_path  +  args.model + str(args.non_linear) + "_norm" +  str(args.normalize) +  "_lr"+ str(args.lr) +
                        '_channels' + str(args.channels) +
                        ".csv", index=False)
