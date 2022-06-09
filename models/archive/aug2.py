import torch as th
import numpy as np
import dgl
import math


def gae_aug(rec, graph, x, feat_drop_rate, mask_prob):

    E = graph.number_of_edges()
    n_node = graph.number_of_nodes()
    n = round_up_to_even(E*mask_prob)
    # extract edge information
    src = graph.edges()[0]
    dst = graph.edges()[1]
    # index of edges that will be removed
    ind = rec[src,dst].argsort()[:n]
    np.random.shuffle(ind.numpy())
    ind1, ind2 = ind.chunk(2)
    # construct a mask
    mask_idx1 = th.ones(E)
    mask_idx2 = th.ones(E)
    mask_idx1[ind1] = 0
    mask_idx2[ind2] = 0
    mask_idx1 = mask_idx1.nonzero()
    mask_idx2 = mask_idx2.nonzero()
    edge_mask1 = mask_idx1.squeeze(1)
    edge_mask2 = mask_idx2.squeeze(1)
    # construct a new graph
    ng1 = dgl.graph([])
    ng2 = dgl.graph([])
    ng1.add_nodes(n_node)
    ng2.add_nodes(n_node)
    # adding edges
    nsrc1 = src[edge_mask1]
    ndst1 = dst[edge_mask1]
    ng1.add_edges(nsrc1, ndst1)
    nsrc2 = src[edge_mask2]
    ndst2 = dst[edge_mask2]
    ng2.add_edges(nsrc2, ndst2)

    ### Edge adding ###
    # add self loop
    graph_temp = graph.add_self_loop()
    adj_temp = graph_temp.adj().to_dense()
    # reverse adjacency matrix
    negadj = abs(adj_temp-1)
    # reverse adjacency matrix to reverse graph object
    negsrc, negdst = th.nonzero(negadj).T
    revg = dgl.graph([])
    revg.add_nodes(n_node)
    revg.add_edges(negsrc, negdst)
    # extract edge information
    revsrc = revg.edges()[0]
    revdst = revg.edges()[1]
    # index of edges that will be removed
    ind = rec[revsrc,revdst].argsort(descending=True)[:n]
    np.random.shuffle(ind.numpy())
    ind1, ind2 = ind.chunk(2)
    # construct a mask
    E = revg.number_of_edges()
    mask_idx1 = th.ones(E)
    mask_idx2 = th.ones(E)
    mask_idx1[ind1] = 0
    mask_idx2[ind2] = 0
    mask_idx1 = mask_idx1.nonzero()
    mask_idx2 = mask_idx2.nonzero()
    edge_mask1 = mask_idx1.squeeze(1)
    edge_mask2 = mask_idx2.squeeze(1)
    # construct a new graph
    nrevg1 = dgl.graph([])
    nrevg2 = dgl.graph([])
    nrevg1.add_nodes(n_node)
    nrevg2.add_nodes(n_node)
    # adding edges
    nsrc1 = revsrc[edge_mask1]
    ndst1 = revdst[edge_mask1]
    nsrc2 = revsrc[edge_mask2]
    ndst2 = revdst[edge_mask2]
    nrevg1.add_edges(nsrc1, ndst1)
    nrevg2.add_edges(nsrc2, ndst2)
    # reverse again
    addadj1 = negadj - nrevg1.adj().to_dense()
    addadj2 = negadj - nrevg2.adj().to_dense()
    addsrc1, adddst1 = th.nonzero(addadj1).T
    addsrc2, adddst2 = th.nonzero(addadj2).T
    ng1.add_edges(addsrc1, adddst1)
    ng2.add_edges(addsrc2, adddst2)

    feat1 = drop_feature(x, feat_drop_rate)
    feat2 = drop_feature(x, feat_drop_rate)

    return ng1, ng2, feat1, feat2

def drop_feature(x, drop_prob):
    drop_mask = th.empty(
        (x.size(1),),
        dtype=th.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def round_up_to_even(f):
    return math.ceil(f / 2.) * 2
