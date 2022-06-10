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


from models.basicVGNAE import *
from models.DeepVGAEX import *
from models.baseline_models import *
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
parser.add_argument('--non_linear', type=str, default='relu')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--result_file', type=str, default="/results/link_prediction_all_")
args = parser.parse_args()

file_path = os.getcwd() + "/results/link_prediction_all_" + str(args.result_file) + '_' + args.dataset +'_normalize' +\
 str(args.normalize) + '_nonlinear' + str(args.non_linear) + '_lr' + str(args.lr) + '.csv'

print(file_path)

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

if args.non_linear == 'relu':
    activation  = torch.nn.ReLU()
else:
    activation = None



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, data, model_name, data_split, y_randoms):
    model.train()
    optimizer.zero_grad()
    all_edge_index = data.edge_index
    loss = model.loss(data.x, y=y_randoms,
                      pos_edge_index=data_split.train_pos_edge_index,
                      all_edge_index=all_edge_index,
                      train_mask=data.train_mask)
    loss.backward()
    optimizer.step()
    return loss


def edge_prediction(embeds, out_dim, pos_edge_index,
                    neg_edge_index, lr=0.01, wd=1e-4,
                    patience = 100, max_epochs=3000):
    logreg = LogReg(embeds.shape[1], out_dim)
    opt = torch.optim.Adam(logreg.parameters(), lr=lr, weight_decay=wd)
    output_activation = nn.Sigmoid()
    last_loss = 1e10
    triggertimes = 0
    best_val_roc = 0
    best_val_ap = 0
    add_neg_samples = True
    loss_fn = torch.nn.BCELoss()
    results = []
    for epoch in range(max_epochs):
        print(epoch)
        logreg.train()
        opt.zero_grad()

        #### 1st alternative:
        logits_temp = logreg(embeds)
        logits = output_activation(torch.mm(logits_temp, logits_temp.t()))
        loss = (loss_fn(logits[pos_edge_index[0,:],pos_edge_index[1,:]], torch.ones(pos_edge_index.shape[1]))+
                    loss_fn(logits[neg_edge_index[0,:],neg_edge_index[1,:]], torch.zeros(neg_edge_index.shape[1])))
        loss.backward(retain_graph=True)
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
            results += [[epoch, val_ap, val_roc, test_ap, test_roc]]
            if current_loss >= last_loss:
                trigger_times += 1
                #print('Trigger Times:', trigger_times)
                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    break
            else:
                #print('trigger times: 0')
                trigger_times = 0
            last_loss = current_loss
    return(logreg, results)

def node_prediction(embeds, out_dim, y, train_mask, test_mask, lr=0.01, wd=1e-4,
                    patience = 100, max_epochs=3000):
    node_classifier = MLP(embeds.shape[1], outdim)
    train_labels = y[train_mask]
    test_labels = y[test_mask]
    optimizer_temp = torch.optim.Adam(node_classifier.parameters(), lr=0.005)
    res_temp = []
    for epoch_temp in range(max_epochs):
        node_classifier.train();
        optimizer_temp.zero_grad();
        out = node_classifier(embeds);
        loss_temp = nn.CrossEntropyLoss()(out[train_mask], train_labels);
        loss_temp.backward()
        optimizer_temp.step()

        preds = torch.argmax(out, dim=1)
        acc_train = torch.sum(preds[train_mask] == train_labels).float() / train_labels.shape[0]
        acc = torch.sum(preds[test_mask] == test_labels).float() / test_labels.shape[0]
        res_temp += [[epoch_temp, loss_temp.cpu().item(), acc_train.item(), acc.item()]]
    return(node_classifier, res_temp)


results =[]
for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
#for training_rate in [0.2]:
    val_ratio = (1.0 - training_rate) / 3
    test_ratio = (1.0 - training_rate) / 3 * 2
    data_split = train_test_split_edges(dataset[0], val_ratio, test_ratio)
    all_edge_index = dataset[0].edge_index
    all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
    all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp) ### make sure they are there, so we don't sample them using negative sampling
    neg_edge_index = negative_sampling(all_edge_index_tmp, num_nodes=data.num_nodes,
                                           num_neg_samples=data_split.train_pos_edge_index.size(1))

    for model in ['VGAEX', 'VGNAE', 'GNAE']:
        #transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio,
        #                            is_undirected=True, split_labels=True)
        if model == 'VGAE':
            alphas = [0., 0.2, 0.5, 0.7, 1.0]
        elif model == 'VGAEX':
            alphas = [1, 10, 50, 100, 500]
        else:
            alphas = [0.1, 0.5, 1.0, 5.0, 10]

        for alpha in alphas:
            for out_channels in [32, 64, 128, 256, 512]:
                train_data, val_data, test_data = transform(data)
                if model == 'VGAE':
                    model = DeepVGAE(data.x.size()[1], out_channels * 2, out_channels,
                                 normalize=True, alpha=alpha,
                                 beta=1.0, non_linearity=activation).to(dev)
                    y_randoms = None
                elif model == 'GNAE':
                    model = GAE(Encoder(data.x.size()[1], out_channels,
                               train_data.pos_edge_label_index, model='GNAE',
                               scaling_factor=alpha)).to(dev)
                    y_randoms = None
                else:
                    model = DeepVGAEX(data.x.size()[1], out_channels * 2, out_channels,
                                     h_dims_reconstructiony = [32, 32],
                                     y_dim=alpha, dropout=0.5,
                                     lambda_y =1.0/alpha, non_linearity=activation).to(dev)
                    w = torch.randn(size= (data.num_features, alpha)).float()
                    y_randoms = torch.mm(data.x, w)
                # move to GPU (if available)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                last_loss = 1e10
                patience = args.patience
                triggertimes = 0

                for epoch in range(1, args.epochs):
                    loss = train(model, optimizer, train_data, model_name, y_randoms)
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
                embeds = model.encode(x, train_pos_edge_index=train_data.pos_edge_label_index)
                nodes_res = node_prediction(embeds, out_dim, y, train_mask, test_mask, lr=0.01, wd=1e-4,
                                                               patience = 100, max_epochs=3000)
                _, _, acc_train, acc = nodes_res[-1]
                results += [[model_name, args.dataset, str(args.non_linear), args.normalize, args.lr, out_channels,
                                      training_rate, val_ratio, test_ratio, alpha, train_auc, train_ap, roc_auc, ap, acc_train, acc epoch, 0, 0]]
                print([model_name, args.dataset, str(args.non_linear), args.normalize, args.lr, out_channels,
                                      training_rate, val_ratio, test_ratio, alpha, train_auc, train_ap, roc_auc, ap, epoch, 0, 0])

                res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                                    'train_rate','val_ratio', 'test_ratio', 'alpha',  'train_auc', 'train_ap',
                                                      'test_auc', 'test_ap', 'accuracy_train', 'accuracy_test', 'epoch',
                                                      'drop_edge_rate', 'drop_feat_rate'])
                res1.to_csv(file_path, index=False)                res1.to_csv(file_path, index=False)


#### Test the CCA approach
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
                    hid_dim = [channels] * n_layers
                    model = CCA_SSG(in_dim, hid_dim, out_dim, use_mlp=False)
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


                    _, res = edge_prediction(embeds, out_dim, pos_edge_index, neg_edge_index, lr=0.01, wd=1e-4,
                                               patience = 100, max_epochs=3000)
                    epoch, val_ap, val_roc, test_ap, test_roc = res[-1]
                    results += [['CCA', args.dataset, str(args.non_linear), args.normalize, args.lr, out_channels,
                                          training_rate, val_ratio, test_ratio, alpha, train_auc, train_ap, roc_auc, ap, acc_train, acc epoch, 0, 0]]
                    print(['CCA', args.dataset, str(args.non_linear), args.normalize, args.lr, out_channels,
                                          training_rate, val_ratio, test_ratio, alpha, train_auc, train_ap, roc_auc, ap, acc_train, acc epoch, 0, 0])

                    res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                                        'train_rate','val_ratio', 'test_ratio', 'alpha',  'train_auc', 'train_ap',
                                                          'test_auc', 'test_ap', 'accuracy_train', 'accuracy_test', 'epoch',
                                                          'drop_edge_rate', 'drop_feat_rate'])
                    res1.to_csv(file_path, index=False)
#### Add the ICA Models


##### Train the ICA model
for training_rate in [0.1]:
    val_ratio = (1.0 - training_rate) / 3
    test_ratio = (1.0 - training_rate) / 3 * 2
    for add_neg_samples in [True]:
        #for lambd in np.logspace(-7, 2, num=1, endpoint=True, base=10.0, dtype=None, axis=0):#np.logspace(-7, 2, num=10, endpoint=True, base=10.0, dtype=None, axis=0):
            for channels in [32]:
            #for channels in [32, 64, 128, 256, 512]:
                    out_dim = channels
                    hid_dim = channels
                    model = ICA_SSG(data.num_features, data.num_features,
                                    hid_dim, out_dim, use_mlp = False,
                                    use_graph=True,
                                    regularize=True)
                    z = model(data)
                    lr1 = 0.01
                    wd1 = 1e-4
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)
                    for epoch in range(args.max_epochs):
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
                    _, res = edge_prediction(embeds, out_dim, pos_edge_index, neg_edge_index, lr=0.01, wd=1e-4,
                                               patience = args.patience, max_epochs=3000)
                    epoch, val_ap, val_roc, test_ap, test_roc = res[-1]
                    nodes_res = node_prediction(embeds, out_dim, y, train_mask, test_mask, lr=0.01, wd=1e-4,
                                                                   patience = 100, max_epochs=3000)
                    _, _, acc_train, acc = nodes_res[-1]

                    results += [[ 'ICA linear', args.dataset, True, True, args.lr, channels,
                                 training_rate, val_ratio, test_ratio, lambd, test_roc,
                                 test_ap, acc_train, acc, epoch, drop_rate_edge, drop_rate_edge ]]

                    res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                                        'train_rate', 'val_ratio', 'test_ratio',
                                                        'test_ratio', 'alpha', 'test_roc', 'test_ap', 'epoch', 'drop_edge_rate', 'drop_feat_rate'])
                    res1.to_csv(file_path, index=False)


##### Train the non linear ICA model
#for training_rate in [0.1, 0.2, 0.4, 0.6, 0.8, 0.85]:
for training_rate in [0.1]:
    val_ratio = (1.0 - training_rate) / 3
    test_ratio = (1.0 - training_rate) / 3 * 2
    for add_neg_samples in [True]:
        #for lambd in np.logspace(-7, 2, num=1, endpoint=True, base=10.0, dtype=None, axis=0):#np.logspace(-7, 2, num=10, endpoint=True, base=10.0, dtype=None, axis=0):
            for channels in [32]:
            #for channels in [32, 64, 128, 256, 512]:
                    latent_dim, = channels
                    aux_dim = dataset.num_classes
                    model = iVAE(latent_dim, data_dim, aux_dim, activation='lrelu', device=device,
                                 n_layers=n_layers, hidden_dim=hidden_dim)
                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience, verbose=True)

                    # training loop

                    print("Training..")
                    epoch = 0
                    model.train()
                    for epoch in range(args.max_epochs):
                        elbo_train = 0
                        optimizer.zero_grad()
                        x = data.x.to(dev)
                        u = torch.nn.functional.one_hot(data.y,
                                                    num_classes=dataset.num_classes).float().to(dev)
                        print(u.shape, x.shape)
                        elbo, z_est = model.elbo(x, u, data.edge_index)
                        elbo.mul(-1).backward()
                        optimizer.step()
                        elbo_train += -elbo.item()
                        #elbo_train /= len(train_loader)
                        scheduler.step(elbo_train)
                        epoch +=1
                        print('epoch {}/{} \tloss: {}'.format(epoch, args.max_epochs, elbo_train))
                    # save model checkpoint after training
                    print("=== Evaluation ===")
                    data = dataset[0]
                    X, U = data.x, u
                    params = {'decoder': decoder_params, 'encoder': encoder_params, 'prior': prior_params}
                    decoder_params, encoder_params, z, prior_params = model(X, U, data.edge_index)
                    embeds = params['encoder'][0].detach()
                    _, res = edge_prediction(embeds, out_dim, pos_edge_index, neg_edge_index, lr=0.01, wd=1e-4,
                                               patience = args.patience, max_epochs=3000)
                    epoch, val_ap, val_roc, test_ap, test_roc = res[-1]
                    nodes_res = node_prediction(embeds, out_dim, y, train_mask, test_mask, lr=0.01, wd=1e-4,
                                                                   patience = 100, max_epochs=3000)
                    _, _, acc_train, acc = nodes_res[-1]

                    results += [[ 'ICA nonlinear', args.dataset, True, True, args.lr, channels,
                                 training_rate, val_ratio, test_ratio, lambd, test_roc,
                                 test_ap, acc_train, acc, epoch, drop_rate_edge, drop_rate_edge ]]

                    res1 = pd.DataFrame(results, columns=['model', 'dataset', 'non-linearity', 'normalize',  'lr', 'channels',
                                                        'train_rate', 'val_ratio', 'test_ratio',
                                                        'test_ratio', 'alpha', 'test_roc', 'test_ap', 'epoch', 'drop_edge_rate', 'drop_feat_rate'])
                    res1.to_csv(file_path, index=False)
