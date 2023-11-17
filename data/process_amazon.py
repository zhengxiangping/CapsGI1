import numpy as np
import csv
from pymetis import part_graph

# Set random seed
seed = 0
np.random.seed(seed)

def partition(adj_raw, n):
    node_num = len(set(adj_raw[0])) + len(set(adj_raw[1]))
    print(node_num)
    adj_list = [[] for _ in range(node_num)]
    for i, j in zip(adj_raw[0], adj_raw[1]):
        if i == j:
            continue
        adj_list[i].append(j)
        adj_list[j].append(i)

    _, ss_labels = part_graph(nparts=n, adjacency=adj_list)

    return ss_labels

# Load data
print("Loading dataset")
# load adjacency matrix and the ground_truth

net = []
with open('amazon_network.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    ttt = 0
    for d in data:
        print(ttt / len(data))
        ttt += 1
        #if d[0] in uu_final:
        net.append([d[0], d[1]])

productID = {}
userID = {}
for key_word in net:
    if key_word[0] not in userID:
        userID[key_word[0]] = len(userID.keys())
    if key_word[1] not in productID:
        productID[key_word[1]] = len(productID.keys())

G = {}
for key_word in net:
    if (userID[key_word[0]], productID[key_word[1]]) not in G.keys():
        G[(userID[key_word[0]], productID[key_word[1]])] = 1
    else:
        G[(userID[key_word[0]], productID[key_word[1]])] += 1

f = open('tmp_Incons.txt','a')
for key in G.keys():
    f.write(str(key[0])+' '+str(key[1])+'\n')
f.close()

adj = np.loadtxt('tmp_Incons.txt')
adj = adj[:, 0: 2]
num_user = len(set(adj[:, 0]))
num_object = len(set(adj[:, 1]))
adj[:, 1] += (np.max(adj[:, 0])+1)
adj = adj.astype('int')
edge_index = adj.T
ss_label = partition(edge_index, 20)
ss_label = np.array(ss_label)

sampled_node1 = np.where(ss_label==0)[0]
sampled_node2 = np.where(ss_label==1)[0]
#sampled_node3 = np.where(ss_label==2)[0]
#sampled_node4 = np.where(ss_label==3)[0]
#sampled_node5 = np.where(ss_label==4)[0]
#sampled_node6 = np.where(ss_label==5)[0]
sampled_node = list(set(list(sampled_node1)+list(sampled_node2)))#+list(sampled_node3)+list(sampled_node4)+list(sampled_node5)+list(sampled_node6)))
sampled_node = np.array(sampled_node)
print(len(sampled_node))

user_dic = {}
prod_dic = {}
f = open('amazonInCons.txt','a')
for i in range(edge_index.shape[1]):
    if (edge_index[0, i] in sampled_node) and (edge_index[1, i] in sampled_node):
        if edge_index[0, i] not in user_dic.keys():
            user_dic[edge_index[0, i]] = len(user_dic.keys())
        if edge_index[1, i] not in prod_dic.keys():
            prod_dic[edge_index[1, i]] = len(prod_dic.keys())
        f.write(str(user_dic[edge_index[0, i]])+' '+str(prod_dic[edge_index[1, i]])+'\n')
f.close()

label = []
tmp_id = []
with open('amazon_gt.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    for d in data:
        if (d[0] in userID.keys()) and (userID[d[0]] in user_dic.keys()) and (d[0] not in tmp_id):
            tmp_id.append(d[0])
            if d[1] == '-1':
                label.append([user_dic[userID[d[0]]], 1])
            else:
                label.append([user_dic[userID[d[0]]], 0])
print(len(label), np.sum(np.array(label)[:, 1]))
np.savetxt('amazon_label.txt', label, fmt="%d")

adj = np.loadtxt('amazonInCons.txt')
adj = adj[:, 0: 2]
num_user = len(set(adj[:, 0]))
num_object = len(set(adj[:, 1]))
adj[:, 1] += num_user
adj = adj.astype('int')
np.savetxt('amazon.txt', adj, fmt="%d")

