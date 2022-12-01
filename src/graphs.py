from collections import defaultdict
from gen_dataset import build_edge_map
import numpy as np


class Graph:
    def __init__(self, num_nodes, tree_labels):
        self.edge_map = build_edge_map()
        self.adj_lists = defaultdict(list)
        for n in range(num_nodes):
            for v in range(num_nodes):
                if n != v:
                    self.adj_lists[n].append(v)
        self.tree_labels = tree_labels

    def weight(self, u, v):
        return 1 - (self.tree_labels[self.edge_map[(u, v)]] if (u,v) in self.edge_map else 0)

    def print(self):
        for u, adj in self.adj_lists.items():
            print(
                str(u) + ': [' + ' , '.join(f'-- {self.weight(u,v):.4f} --> {v}' for v in adj) + ']')


def remove_min(priorities):
    min_priority_vertex = None
    min_priority = 2
    for v, (priority, _) in priorities.items():
        if priority < min_priority:
            min_priority = priority
            min_priority_vertex = v
    pred = priorities[min_priority_vertex][1]
    del priorities[min_priority_vertex]
    return min_priority_vertex, pred


# O(EV)
def prims(graph):
    priorities = dict()
    tree = np.zeros(45)
    for v in graph.adj_lists:
        priorities[v] = [2, None]
    priorities[0] = [0, None]
    print(priorities)
    while len(priorities) > 0:
        u, up = remove_min(priorities)
        if up is not None:
            tree[graph.edge_map[(up, u)]] = 1
        for v in graph.adj_lists[u]:
            if v in priorities and graph.weight(u, v) < priorities[v][0]:
                priorities[v][0] = graph.weight(u, v)
                priorities[v][1] = u
    return tree
