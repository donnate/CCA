from numbers import Number
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from models.baseline_models import MLP, LogReg


def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


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


# def train(model, optimizer, train_data, model_name, y_randoms):
#     model.train()
#     optimizer.zero_grad()
#     loss = model.loss(train_data.x, y=y_randoms,
#                       pos_edge_index=train_data.pos_edge_label_index,
#                       neg_edge_index=train_data.neg_edge_label_index,
#                       train_mask=train_data.train_mask)
#     loss.backward()
#     optimizer.step()
#     return loss


def edge_prediction(embeds, out_dim, train_data, test_data, val_data,
                    lr=0.01, wd=1e-4,
                    patience = 30, max_epochs=3000):
    logreg = LogReg(embeds.shape[1], out_dim)
    opt = torch.optim.Adam(logreg.parameters(), lr=lr, weight_decay=wd)
    output_activation = torch.nn.Sigmoid()
    last_loss = 1e10
    triggertimes = 0
    best_val_roc = 0
    best_val_ap = 0
    add_neg_samples = True
    loss_fn = torch.nn.BCELoss()
    results = []
    pos_edge_index = train_data.pos_edge_label_index
    neg_edge_index = train_data.neg_edge_label_index
    for epoch in range(max_epochs):
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
            try:
                train_roc, train_ap = get_scores(train_data.pos_edge_label_index, train_data.neg_edge_label_index, logits)
            except:
                train_roc, train_ap = np.nan, np.nan

            if np.isnan(val_roc):
                break
            if (np.isnan(val_roc) ==False) & (val_roc >= best_val_roc):
                best_val_roc = val_roc

            current_loss = val_roc
            results += [[epoch, val_ap, val_roc, test_ap, test_roc, train_ap, train_roc ]]
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
                    patience = 30, max_epochs=3000):
    #input_dim, hidden_dim, output_dim, n_layers=2, activation='relu', slope=.1, device='cpu', use_bn=False
    node_classifier = MLP(embeds.shape[1], embeds.shape[1], out_dim,  n_layers=2)
    train_labels = y[train_mask]
    test_labels = y[test_mask]
    optimizer_temp = torch.optim.Adam(node_classifier.parameters(), lr=0.005)
    res_temp = []
    for epoch_temp in range(max_epochs):
        node_classifier.train();
        optimizer_temp.zero_grad();
        out = node_classifier(embeds);
        loss_temp = torch.nn.CrossEntropyLoss()(out[train_mask], train_labels);
        loss_temp.backward()
        optimizer_temp.step()

        preds = torch.argmax(out, dim=1)
        acc_train = torch.sum(preds[train_mask] == train_labels).float() / train_labels.shape[0]
        acc = torch.sum(preds[test_mask] == test_labels).float() / test_labels.shape[0]
        res_temp += [[epoch_temp, loss_temp.cpu().item(), acc_train.item(), acc.item()]]
    return(node_classifier, res_temp)
