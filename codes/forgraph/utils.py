from config import *
import numpy as np
import pickle as pkl
import networkx as nx
from scipy.sparse import coo_matrix
import sys

def get_graph_data(dataset):
    pri = './dataset/'+dataset+'/'+dataset+'_'

    file_edges = pri+'A.txt'
    # file_edge_labels = pri+'edge_labels.txt'
    file_edge_labels = pri+'edge_gt.txt'
    file_graph_indicator = pri+'graph_indicator.txt'
    file_graph_labels = pri+'graph_labels.txt'
    file_node_labels = pri+'node_labels.txt'

    edges = np.loadtxt( file_edges,delimiter=',').astype(np.int32)
    try:
        edge_labels = np.loadtxt(file_edge_labels,delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use edge label 0')
        edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

    graph_indicator = np.loadtxt(file_graph_indicator,delimiter=',').astype(np.int32)
    graph_labels = np.loadtxt(file_graph_labels,delimiter=',').astype(np.int32)

    try:
        node_labels = np.loadtxt(file_node_labels,delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use node label 0')
        node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

    graph_id = 1
    starts = [1]
    node2graph = {}
    for i in range(len(graph_indicator)):
        if graph_indicator[i]!=graph_id:
            graph_id = graph_indicator[i]
            starts.append(i+1)
        node2graph[i+1]=len(starts)-1
    # print(starts)
    # print(node2graph)
    graphid  = 0
    edge_lists = []
    edge_label_lists = []
    edge_list = []
    edge_label_list = []
    for (s,t),l in list(zip(edges,edge_labels)):
        sgid = node2graph[s]
        tgid = node2graph[t]
        if sgid!=tgid:
            print('edges connecting different graphs, error here, please check.')
            print(s,t,'graph id',sgid,tgid)
            exit(1)
        gid = sgid
        if gid !=  graphid:
            edge_lists.append(edge_list)
            edge_label_lists.append(edge_label_list)
            edge_list = []
            edge_label_list = []
            graphid = gid
        start = starts[gid]
        edge_list.append((s-start,t-start))
        edge_label_list.append(l)

    edge_lists.append(edge_list)
    edge_label_lists.append(edge_label_list)

    # node labels
    node_label_lists = []
    graphid = 0
    node_label_list = []
    for i in range(len(node_labels)):
        nid = i+1
        gid = node2graph[nid]
        # start = starts[gid]
        if gid!=graphid:
            node_label_lists.append(node_label_list)
            graphid = gid
            node_label_list = []
        node_label_list.append(node_labels[i])
    node_label_lists.append(node_label_list)

    return edge_lists, graph_labels, edge_label_lists, node_label_lists


def load_real_dataset(dataset):

    try:
        with open('./'+dataset+'.pkl','rb') as fin:
            return pkl.load(fin)
    except:

        edge_lists, graph_labels, edge_label_lists, node_label_lists = get_graph_data(dataset)


        graph_labels[graph_labels == -1] = 0

        max_node_nmb = np.max([len(node_label) for node_label in node_label_lists]) + 1  # add nodes for each graph

        edge_label_nmb = np.max([np.max(l) for l in edge_label_lists]) + 1
        node_label_nmb = np.max([np.max(l) for l in node_label_lists]) + 1

        for gid in range(len(edge_lists)):
            node_nmb = len(node_label_lists[gid])
            for nid in range(node_nmb, max_node_nmb):
                edge_lists[gid].append((nid, nid))  # add self edges
                node_label_lists[gid].append(node_label_nmb)  # the label of added node is node_label_nmb
                edge_label_lists[gid].append(edge_label_nmb)

        adjs = []
        for edge_list in edge_lists:
            row = np.array(edge_list)[:, 0]
            col = np.array(edge_list)[:, 1]
            data = np.ones(row.shape)
            adj = coo_matrix((data, (row, col))).toarray()
            if args.normadj:
                degree = np.sum(adj, axis=0, dtype=float).squeeze()
                degree[degree == 0] = 1
                sqrt_deg = np.diag(1.0 / np.sqrt(degree))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            adjs.append(np.expand_dims(adj, 0))

        labels = graph_labels

        adjs = np.concatenate(adjs, 0)
        labels = np.array(labels).astype(int)
        feas = []

        for node_label in node_label_lists:
            fea = np.zeros((len(node_label), node_label_nmb + 1))
            rows = np.arange(len(node_label))
            fea[rows, node_label] = 1
            fea = fea[:, :-1]  # remove the added node feature

            if node_label_nmb < 3:
                const_features = np.ones([fea.shape[0], 10])
                fea = np.concatenate([fea, const_features], -1)
            feas.append(fea)

        feas = np.array(feas)

        # print(max_node_nmb)

        b = np.zeros((labels.size, labels.max() + 1))
        b[np.arange(labels.size), labels] = 1
        labels = b
        with open('./'+dataset+'.pkl','wb') as fout:
            pkl.dump((adjs, feas,labels),fout)
        return adjs, feas,labels
