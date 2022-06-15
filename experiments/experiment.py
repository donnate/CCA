import sys, os
import os.path as osp
import numpy as np
sys.path.append('/scratch/midway3/cdonnat/CCA')

import argparse
import pandas as pd
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


from models.basicVGNAE import *
from models.DeepVGAEX import *
from models.baseline_models import *
from aug import *
from models.cca import CCA_SSG
from models.ica_gnn import GraphICA, iVGAE, random_permutation
from train_utils import *

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
parser.add_argument('--model', type=str, default='VGAEX')
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--normalize', type=parse_boolean, default=True)
parser.add_argument('--non_linear', type=str, default='relu')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--max_epoch_eval', type=int, default=2000)
parser.add_argument('--result_file', type=str, default="/results/n_experiments_")
args = parser.parse_args()

MAX_EPOCH_EVAL = args.max_epoch_eval

file_path = os.getcwd() + str(args.result_file) + args.model + '_' + args.dataset +'_normalize' +\
 str(args.normalize) + '_nonlinear' + str(args.non_linear) + '_lr' + str(args.lr) + '.csv'

print(file_path)
path = '/scratch/midway3/cdonnat/CCA/data'
if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(root='/scratch/midway3/cdonnat/CCA/data/Planetoid', name=args.dataset, transform=NormalizeFeatures())
    data =  dataset[0]
if args.dataset in ['cs', 'physics']:
    dataset = Coauthor(path, args.dataset, 'public')
    data = dataset[0]
    data = T.NormalizeFeatures()(data)
if args.dataset in ['computers', 'photo']:
    dataset = Amazon(path, args.dataset, 'public')
    data = dataset[0]
    data = T.NormalizeFeatures()(data)

if args.non_linear == 'relu':
    activation  = torch.nn.ReLU()
else:
    activation = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.model == 'VGNAE':
    alphas = [0.]
elif args.model == 'VGAEX':
    alphas = [1, 10, 50, 100, 500]
else:
    alphas = [0]

n_layers = [1, 2, 3, 4, 5]

results =[]
if args.model in ['VGNAE', 'VGAEX']:
    for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
        val_ratio = (1.0 - training_rate) / 3
        test_ratio = (1.0 - training_rate) / 3 * 2
        transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                    is_undirected=True, split_labels=True)
        train_data, val_data, test_data = transform(data)
        all_edge_index = dataset[0].edge_index
        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp) ### make sure they are there, so we don't sample them using negative sampling
        neg_edge_index = negative_sampling(all_edge_index_tmp, num_nodes=data.num_nodes,
                                           num_neg_samples=train_data.pos_edge_label_index.size(1))

        #for alpha in alphas:
        for alpha in alphas:
            for n_lay in n_layers:
                #for out_channels in [ 32]:
                for out_channels in [32, 64, 128, 256, 512]:
                    if args.model == 'VGNAE':
                        model = DeepVGAE(data.x.size()[1], out_channels * 2, out_channels,
                                         n_layers=n_lay, normalize=True,
                                         activation=args.non_linear).to(device)
                        y_randoms = None
                    else:
                        model = DeepVGAEX(data.x.size()[1], out_channels * 2, out_channels,
                                         n_layers=n_lay, normalize=True,
                                         h_dims_reconstructiony = [out_channels, out_channels],
                                         y_dim=alpha, dropout=0.5,
                                         lambda_y =0.5/alpha, activation=args.non_linear).to(device)
                        w = torch.randn(size= (data.num_features, alpha)).float()
                        y_randoms = torch.mm(data.x, w)
                    # move to GPU (if available)
                    model = model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                    last_loss = 0
                    triggertimes = 0

                    for epoch in range(1, args.epochs):
                        model.train()
                        optimizer.zero_grad()
                        loss = model.loss(train_data.x, y=y_randoms,
                                          pos_edge_index=train_data.pos_edge_label_index,
                                          neg_edge_index=train_data.neg_edge_label_index,
                                          train_mask=train_data.train_mask)
                        loss.backward()
                        optimizer.step()
                        if epoch == 50: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/2)
                        if epoch == 100: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/4)
                        if epoch == 150: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/8)
                        if epoch % 5 == 0:
                            loss = float(loss)
                            train_auc, train_ap = model.single_test(data.x,
                                                train_data.pos_edge_label_index,
                                                train_data.pos_edge_label_index,
                                                train_data.neg_edge_label_index)
                            roc_auc, ap = model.single_test(data.x,
                                                train_data.pos_edge_label_index,
                                                test_data.pos_edge_label_index,
                                                test_data.neg_edge_label_index)
                            print('Epoch: {:03d}, LOSS: {:.4f}, AUC(train): {:.4f}, AP(train): {:.4f}  AUC(test): {:.4f}, AP(test): {:.4f}'.format(epoch, loss, train_auc, train_ap, roc_auc, ap))

                            #### Add early stopping to prevent overfitting
                            out  = model.single_test(data.x,
                                                train_data.pos_edge_label_index,
                                                val_data.pos_edge_label_index,
                                                val_data.neg_edge_label_index)
                            current_loss = np.mean(out)
                            if current_loss <= last_loss:
                                trigger_times += 1
                                #print('Trigger Times:', trigger_times)
                                #if triggertimes == 2: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/2)
                                #if triggertimes == 6: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/4)
                                #if triggertimes == 10: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/8)
                                if trigger_times >= args.patience:
                                    #print('Early stopping!\nStart to test process.')
                                    break
                            else:
                                #print('trigger times: 0')
                                trigger_times = 0
                                last_loss = current_loss
                    embeds = model.encode(train_data.x, edge_index=train_data.pos_edge_label_index)
                    _, nodes_res = node_prediction(embeds.detach(),
                                                   dataset.num_classes, data.y,
                                                   data.train_mask, data.test_mask,
                                                   lr=0.01, wd=1e-4, patience = 100,
                                                   max_epochs=3000)
                    acc_train, acc = nodes_res[-1][2], nodes_res[-1][3]
                    results += [[args.model, args.dataset, str(args.non_linear), args.normalize, args.lr, out_channels,
                                          training_rate, val_ratio, test_ratio, n_lay, alpha, train_auc, train_ap,
                                          roc_auc, ap, acc_train, acc, epoch, 0, 0]]
                    res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                                          'train_rate','val_ratio', 'test_ratio', 'n_layers', 'alpha',  'train_auc', 'train_ap',
                                                          'test_auc', 'test_ap', 'accuracy_train', 'accuracy_test', 'epoch',
                                                          'drop_edge_rate', 'drop_feat_rate'])
                    res1.to_csv(file_path, index=False)
elif args.model == 'CCA':
    #### Test the CCA approach
    print("CCA_SSG")
    data = dataset[0]
    in_dim = data.num_features
    N = data.num_nodes

    ##### Train the CCA model
    for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
    #for training_rate in [0.1]:
        val_ratio = (1.0 - training_rate) / 3
        test_ratio = (1.0 - training_rate) / 3 * 2
        transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                        is_undirected=True, split_labels=True)
        train_data, val_data, test_data = transform(data)
        for n in n_layers:
            #for lambd in [1.]:
            for lambd in np.logspace(-7, 2, num=1, endpoint=True, base=10.0, dtype=None, axis=0):#np.logspace(-7, 2, num=10, endpoint=True, base=10.0, dtype=None, axis=0):
                for channels in [32, 64, 128, 256, 512]:
                #for channels in [32]:
                    for drop_rate_edge in [0.01, 0.05, 0.1, 0.2, 0,3, 0.4, 0.5, 0.7]:
                    #for drop_rate_edge in [0.01]:
                        out_dim = channels
                        hid_dim = [channels] * n
                        model = CCA_SSG(in_dim, hid_dim, out_dim, use_mlp=False)
                        wd1 = 1e-4
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                                     weight_decay=wd1)
                        #for epoch in range(300):
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
                        logreg = LogReg(embeds.shape[1], adj_train.shape[1])
                        opt = torch.optim.Adam(logreg.parameters(), lr=args.lr, weight_decay=1e-4)

                        _, res = edge_prediction(embeds.detach(), embeds.shape[1],
                                                 train_data, test_data, val_data,
                                                 lr=0.01, wd=1e-4,
                                                 patience = args.patience,
                                                 max_epochs=MAX_EPOCH_EVAL)
                        val_ap, val_roc, test_ap, test_roc, train_ap, train_roc = res[-1][1], res[-1][2], res[-1][3], res[-1][4], res[-1][5], res[-1][6]
                        _, nodes_res = node_prediction(embeds.detach(),
                                                       dataset.num_classes, data.y,
                                                       data.train_mask, data.test_mask,
                                                       lr=0.01, wd=1e-4,
                                                       patience = args.patience,
                                                       max_epochs=MAX_EPOCH_EVAL)

                        acc_train, acc = nodes_res[-1][2], nodes_res[-1][3]

                        results += [['CCA', args.dataset, str(args.non_linear),
                                     args.normalize, args.lr, channels,
                                     training_rate, val_ratio, test_ratio,
                                     n, lambd, train_roc, train_ap,
                                     test_roc, test_ap, acc_train, acc, epoch, 0, 0]]
                        print(['CCA', args.dataset, str(args.non_linear),
                               args.normalize, args.lr, channels,
                               training_rate, val_ratio, test_ratio,
                               n, lambd, train_roc, train_ap,
                               test_roc, test_ap, acc_train, acc, epoch, 0, 0])

                        res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                                            'train_rate','val_ratio', 'test_ratio', 'n_layers', 'lambd',  'train_auc', 'train_ap',
                                                              'test_auc', 'test_ap', 'accuracy_train', 'accuracy_test', 'epoch',
                                                              'drop_edge_rate', 'drop_feat_rate'])
                        res1.to_csv(file_path, index=False)
elif args.model == 'ICA':
    print("ICA 1")
    criterion = torch.nn.CrossEntropyLoss()
    ##### Train the ICA model
    for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
        val_ratio = (1.0 - training_rate) / 3
        test_ratio = (1.0 - training_rate) / 3 * 2
        transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                        is_undirected=True, split_labels=True)
        train_data, val_data, test_data = transform(data)
        for n in n_layers:
            #for lambd in np.logspace(-7, 2, num=1, endpoint=True, base=10.0, dtype=None, axis=0):#np.logspace(-7, 2, num=10, endpoint=True, base=10.0, dtype=None, axis=0):
                #for channels in [32]:
                for channels in [32, 64, 128, 256, 512]:
                        model = GraphICA(data.num_features,
                                        channels, channels, use_mlp = False,
                                        use_graph=True,
                                        regularize=True)
                        z = model(data)
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
                        for epoch in range(args.epochs):
                            model.train()
                            optimizer.zero_grad()
                            new_data1 = random_permutation(data)

                            z = model(data)
                            z_fake = model(new_data1)
                            loss_pos = criterion(z, torch.ones(z.shape[0]).long())
                            loss_neg = criterion(z_fake, torch.zeros(z_fake.shape[0]).long())
                            #neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

                            loss = loss_pos + loss_neg

                            loss.backward()
                            optimizer.step()
                            preds  = torch.argmax(z, 1)
                            preds0  = torch.argmax(z_fake, 1)
                            acc = 0.5*((preds==1).sum()/preds.shape[0]).item() +  0.5* ((preds0==0).sum()/preds0.shape[0]).item()
                            print('Epoch={:03d}, loss={:.4f}, acc={:.4f}'.format(epoch, loss.item(), acc))


                        print("=== Evaluation ===")
                        data = dataset[0]
                        embeds = model.get_embedding(train_data).detach()
                        _, res = edge_prediction(embeds, embeds.shape[1], train_data, test_data, val_data,
                                            lr=0.01, wd=1e-4,
                                            patience = args.patience, max_epochs=MAX_EPOCH_EVAL)
                        epoch, val_ap, val_roc, test_ap, test_roc, train_ap, train_roc  = res[-1]
                        _, nodes_res = node_prediction(embeds, dataset.num_classes, data.y, data.train_mask, data.test_mask, lr=0.01, wd=1e-4,
                                                                       patience = args.patience, max_epochs=MAX_EPOCH_EVAL)
                        _, _, acc_train, acc = nodes_res[-1]

                        results += [[ 'ICA linear', args.dataset, args.non_linear, True, args.lr, channels,
                                     training_rate, val_ratio, test_ratio, n, train_roc, train_ap, test_roc,
                                     test_ap, acc_train, acc, epoch, None, None]]

                        res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                        'train_rate','val_ratio', 'test_ratio', 'n_layers',  'train_auc', 'train_ap',
                                          'test_auc', 'test_ap', 'accuracy_train', 'accuracy_test', 'epoch',
                                          'drop_edge_rate', 'drop_feat_rate'])
                        res1.to_csv(file_path, index=False)

else:
    ##### Train the non linear ICA model
    print("ICA non linear")
    for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
    #for training_rate in [0.1]:
        val_ratio = (1.0 - training_rate) / 3
        test_ratio = (1.0 - training_rate) / 3 * 2
        transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
                                        is_undirected=True, split_labels=True)
        train_data, val_data, test_data = transform(data)
        for n in n_layers:
            #for lambd in np.logspace(-7, 2, num=1, endpoint=True, base=10.0, dtype=None, axis=0):#np.logspace(-7, 2, num=10, endpoint=True, base=10.0, dtype=None, axis=0):
                #for channels in [32]:
                for channels in [32, 64, 128, 256, 512]:
                        aux_dim = dataset.num_classes
                        model = iVGAE(latent_dim=channels, data_dim=data.num_features,
                                      aux_dim=dataset.num_classes, activation=args.non_linear,
                                      device=device, n_layers=2, hidden_dim = channels)
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience, verbose=True)

                        # training loop

                        print("Training..")
                        model.train()
                        x = data.x.to(device)
                        u = torch.nn.functional.one_hot(data.y,
                                    num_classes=dataset.num_classes).float().to(device)

                        for epoch in range(args.epochs):
                            elbo_train = 0

                            optimizer.zero_grad()
                            elbo, z_est = model.elbo(x, u, data.edge_index)
                            elbo.mul(-1).backward()
                            optimizer.step()
                            elbo_train += -elbo.item()
                            #elbo_train /= len(train_loader)
                            #scheduler.step(elbo_train)
                            print('epoch {}/{} \tloss: {}'.format(epoch, args.epochs, elbo_train))
                        # save model checkpoint after training
                        print("=== Evaluation ===")
                        data = dataset[0]
                        Xt, Ut = data.x, u
                        decoder_params, encoder_params, z, prior_params = model(Xt, Ut, data.edge_index)
                        params = {'decoder': decoder_params, 'encoder': encoder_params, 'prior': prior_params}
                        embeds = params['encoder'][0].detach()
                        _, res = edge_prediction(embeds, embeds.shape[1], train_data, test_data, val_data,
                                            lr=0.01, wd=1e-4,
                                            patience = args.patience, max_epochs=MAX_EPOCH_EVAL)
                        epoch, val_ap, val_roc, test_ap, test_roc, train_ap, train_roc = res[-1]
                        _, nodes_res = node_prediction(embeds, dataset.num_classes, data.y, data.train_mask, data.test_mask,
                                                       lr=0.01, wd=1e-4,
                                                       patience = args.patience, max_epochs=MAX_EPOCH_EVAL)
                        acc_train, acc = nodes_res[-1][2], nodes_res[-1][3]

                        results += [[ 'ICA nonlinear', args.dataset,
                                      args.non_linear, True, args.lr, channels,
                                     training_rate, val_ratio, test_ratio,
                                     n, train_roc, train_ap, test_roc,
                                     test_ap, acc_train, acc, epoch,
                                     None, None ]]

                        res1 = pd.DataFrame(results, columns=['model', 'dataset',
                                                              'non-linearity', 'normalize',
                                                              'lr', 'channels',
                                                              'train_rate','val_ratio',
                                                              'test_ratio', 'n_layers',
                                                              'train_auc', 'train_ap',
                                                              'test_auc', 'test_ap',
                                                              'accuracy_train', 'accuracy_test',
                                                              'epoch', 'drop_edge_rate',
                                                              'drop_feat_rate'])
                        res1.to_csv(file_path, index=False)
