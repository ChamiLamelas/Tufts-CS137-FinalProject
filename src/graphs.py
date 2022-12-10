from collections import defaultdict
from gen_dataset import build_edge_map
import numpy as np
import torch

class Graph:
    def __init__(self, digit_labels, tree_labels):
        self.edge_map = build_edge_map()
        self.adj_lists = defaultdict(list)
        num_nodes = digit_labels.size()[0] - 1
        while num_nodes > 0 and digit_labels[num_nodes - 1] == 0:
            num_nodes -= 1
        for n in range(num_nodes):
            for v in range(num_nodes):
                if n != v:
                    self.adj_lists[n].append(v)
        self.tree_labels = tree_labels

    def weight(self, u, v):
        return 1 - (self.tree_labels[self.edge_map[(u, v)]] if (u, v) in self.edge_map else 0)

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
    tree = np.zeros(45, dtype=np.int32)
    for v in graph.adj_lists:
        priorities[v] = [2, None]
    priorities[0] = [0, None]
    while len(priorities) > 0:
        u, up = remove_min(priorities)
        if up is not None:
            tree[graph.edge_map[(up, u)]] = 1
        for v in graph.adj_lists[u]:
            if v in priorities and graph.weight(u, v) < priorities[v][0]:
                priorities[v][0] = graph.weight(u, v)
                priorities[v][1] = u
    return tree

def main():
    g = Graph(torch.Tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]), torch.Tensor([0.9, 0.4] + [0] * 43))
    g.print()
    tree = prims(g)
    print(tree)

    # change 4,9 if we're interested in indices for other indexes
    edge_map = build_edge_map()
    print(edge_map[(4,9)])

if __name__ == '__main__':
    main()