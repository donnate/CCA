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
from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops,
     to_dense_adj
)


from VGNAE import *
from GeneralizedVGAE import *
from model import *
from aug import *

def parse_boolean(value):
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--normalize', type=parse_boolean, default=True)
parser.add_argument('--non_linear', type=parse_boolean, default=True)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--result_file', type=str, default="/results/link_prediction_all_")
args = parser.parse_args()

file_path = os.getcwd() + args.result_file + args.dataset +'_normalize' +\
 str(args.normalize) + '_nonlinear' + str(args.non_linear) + '_lr' + str(args.lr) + '.csv'



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
    for model in ['Gen-VGNAE', 'Gen-GNAE', 'VGNAE', 'GNAE']:
        transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                    is_undirected=True, split_labels=True)
        if model in ['Gen-VGNAE', 'Gen-GNAE']:
            alphas = np.arange(0,1.1, 0.1)
        else:
            alphas = [1.0, 1.5, 1.8, 2.0, 5.0, 10.]
        for alpha in alphas:
            for out_channels in [32, 64, 128, 256, 512]:
                train_data, val_data, test_data = transform(data)
                if model == 'Gen-GNAE':
                    model = GAE(GCNEncoder(data.x.size()[1], out_channels, norm=True, alpha=alpha,
                    beta=1.0, non_linearity=args.non_linear))
                elif model == 'Gen-VGNAE':
                    model = VGAE(VariationalGCNEncoder(data.x.size()[1], out_channels, norm=True, alpha=alpha,
                                             beta=1.0, non_linearity=args.non_linear))
                elif model == 'GNAE':
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
                    loss = train(model, optimizer, train_data, model)
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
                results += [[model, args.dataset, args.non_linear, args.normalize, args.lr, out_channels,
                                      training_rate, val_ratio, test_ratio, alpha, auc, ap, epoch, 0, 0, False]]
                res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                                      'train_rate', 'val_ratio',
                                                      'test_ratio', 'alpha', 'auc', 'ap', 'epoch','drop_edge_rate', 'drop_feat_rate', 'add_neg_samples'])
                res1.to_csv(file_path, index=False)


n_layers = 2
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0]
in_dim = data.num_features
N = data.num_nodes

from sklearn.metrics import roc_auc_score, average_precision_score
def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


# In[4]:


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for i in range(edges_pos.shape[1]):
        preds.append(sigmoid(adj_rec[edges_pos[0,i], edges_pos[1,i]].item()))

    preds_neg = []
    for i in range(edges_neg.shape[1]):
        preds_neg.append(sigmoid(adj_rec[edges_neg[0,i], edges_neg[1,i]].item()))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


##### Train the CCA model
for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
    val_ratio = (1.0 - training_rate) / 3
    test_ratio = (1.0 - training_rate) / 3 * 2
    for add_neg_samples in [True, False]:
        for lambd in np.logspace(-7, 2, num=1, endpoint=True, base=10.0, dtype=None, axis=0):#np.logspace(-7, 2, num=10, endpoint=True, base=10.0, dtype=None, axis=0):
            for channels in [32, 64, 128, 256, 512]:
                for drop_rate_edge in [0.01, 0.05, 0.1, 0,15, 0.2, 0.25, 0,3, 0.4, 0.5, 0.7]:
                    out_dim = channels
                    hid_dim = channels
                    model = CCA_SSG(in_dim, hid_dim, out_dim, n_layers, use_mlp=False)
                    lr1 = 0.005
                    wd1 = 0
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)
                    for epoch in range(300):
                        model.train()
                        optimizer.zero_grad()

                        dfr = drop_rate_edge
                        der =drop_rate_edge

                        new_data1 = random_aug(data, dfr , der )
                        new_data2 = random_aug(data, dfr, der)

                        z1, z2 = model(new_data1, new_data2)

                        c = torch.mm(z1.T, z2)
                        c1 =torch.mm(z1.T, z1)
                        c2 = torch.mm(z2.T, z2)

                        c = c / N
                        c1 = c1 / N
                        c2 = c2 / N

                        loss_inv = -torch.diagonal(c).sum()
                        iden = torch.tensor(np.eye(c.shape[0]))
                        loss_dec1 = (iden - c1).pow(2).sum()
                        loss_dec2 = (iden - c2).pow(2).sum()

                        loss = loss_inv + lambd * (loss_dec1 + loss_dec2)

                        loss.backward()
                        optimizer.step()

                        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

                    # In[ ]:


                    print("=== Evaluation ===")
                    data = dataset[0]
                    transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                                is_undirected=True, split_labels=True)
                    train_data, val_data, test_data = transform(dataset[0])
                    embeds = model.get_embedding(train_data)
                    adj_train = to_dense_adj(train_data.edge_index,  max_num_nodes=N)
                    adj_train = adj_train[0]
                    #print(embeds.shape, adj_train.shape)
                    weight_tensor, norm = compute_loss_para(adj_train)
                    logreg = LogReg(embeds.shape[1], adj_train.shape[1])
                    lr2 = args.lr
                    wd2 = 1e-4
                    opt = torch.optim.Adam(logreg.parameters(), lr=lr2, weight_decay=wd2)

                    loss_fn = F.binary_cross_entropy
                    output_activation = nn.Sigmoid()

                    last_loss = 1e10
                    patience = args.patience
                    triggertimes = 0
                    best_val_roc = 0
                    best_val_ap = 0

                    for epoch in range(args.epochs):
                        logreg.train()
                        opt.zero_grad()

                        #### 1st alternative:
                        logits_temp = logreg(embeds)
                        logits = output_activation(torch.mm(logits_temp, logits_temp.t()))
                        #print(logits.view(-1).shape, adj_train.view(-1).shape, weight_tensor.shape)


                        # pdb.set_trace()
                        #z = torch.tensor([logits[e[0], e[1]] for e in pos_edge_index.T]+ [logits[e[0], e[1]] for e in neg_edge_index.T])
                        #true_z = torch.from_numpy(np.hstack([np.ones(pos_edge_index.shape[1]), np.zeros(neg_edge_index.shape[1])])).float()
                        #loss = norm * loss_fn(z, true_z)
                        if add_neg_samples == False:
                            loss = norm*loss_fn(logits.view(-1), adj_train.view(-1), weight = weight_tensor)
                        else:
                            it = 0
                            pos_edge_index, _ = remove_self_loops(train_data.pos_edge_label_index)
                            pos_edge_index, _ = add_self_loops(pos_edge_index)
                            neg_edge_index = negative_sampling(pos_edge_index, N)
                            for e in pos_edge_index:
                                if it == 0:
                                    loss = -norm * torch.log(logits[e[0], e[1]])
                                else:
                                    loss += -norm * torch.log(logits[e[0], e[1]])
                                it += 1
                            for e in neg_edge_index:
                                    loss += -norm * torch.log(1-logits[e[0], e[1]])

                        # neg_loss = -torch.log(1 - torch.tensor([logits[e[0], e[1]] for e in pos_edge_index.T])).mean()
                        #


                        loss.backward()
                        opt.step()


                        logreg.eval()
                        with torch.no_grad():
                            try:
                                val_roc, val_ap = get_scores(val_data.pos_edge_label_index, val_data.neg_edge_label_index, logits)
                            except:
                                val_roc, val_ap  = np.nan, np.nan
                            try:
                                test_roc, test_ap = get_scores(test_data.pos_edge_label_index, test_data.neg_edge_label_index, logits)
                            except:
                                test_roc, test_ap = np.nan, np.nan

                            if np.isnan(val_roc):
                                break
                            if (np.isnan(val_roc) ==False) & (val_roc >= best_val_roc):
                                best_val_roc = val_roc

                            current_loss = val_roc
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

                        print('Epoch:{}, val_ap:{:.4f}, val_roc:{:4f}, test_ap:{:4f}, test_roc:{:4f}'.format(epoch, val_ap, val_roc, test_ap, test_roc))
                        print('Linear evaluation AP:{:.4f}'.format(val_ap))
                        print('Linear evaluation ROC:{:.4f}'.format(val_roc))
                        results += [[ 'CCA', args.dataset, True, True, args.lr, channels,
                                              training_rate, val_ratio, test_ratio, lambd, test_roc,
                                              test_ap, epoch, drop_rate_edge, drop_rate_edge, add_neg_samples]]
                        res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                                              'train_rate', 'val_ratio',
                                                              'test_ratio', 'alpha', 'auc', 'ap', 'epoch', 'drop_edge_rate', 'drop_feat_rate', 'add_neg_samples'])
                        res1.to_csv(file_path, index=False)
