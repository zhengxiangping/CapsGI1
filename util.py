import numpy as np
from sklearn.model_selection import StratifiedKFold

def load_data(datasets, num_folds):
    # load the adjacency
    adj = np.loadtxt('./data/'+datasets+'.txt')
    num_user = len(set(adj[:, 0]))
    num_object = len(set(adj[:, 1]))
    adj = adj.astype('int')

    #adj = [(18257,2)]

    nb_nodes = np.max(adj) + 1
    print(nb_nodes)


    edge_index = adj.T
    print('Load the edge_index done!')
    
    # load the user label
    label = np.loadtxt('./data/'+datasets+'_label.txt')
    y = label[:, 1]

    #np.sum(y) 217  len(y) 8227

    print('Ratio of fraudsters: ', np.sum(y) / len(y))
    print('Number of edges: ', edge_index.shape[1])
    print('Number of users: ', num_user)
    print('Number of objects: ', num_object)
    print('Number of nodes: ', nb_nodes)

    # split the train_set and validation_set
    split_idx = []
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    for (train_idx, test_idx) in skf.split(y, y):
        # print('train_idx',len(train_idx),train_idx) # 7404
        # print('test_idx', len(test_idx),test_idx)   # 823

        split_idx.append((train_idx, test_idx))


   
    # load initial features
    feats = np.load('./features/'+datasets+'_feature64.npy')

    # feats (9227, 64)

    # print('feats', feats.shape, feats)


    return edge_index, feats, split_idx, label, nb_nodes

