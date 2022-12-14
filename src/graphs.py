# Authors: Chami Lamelas, Ryan Polhemus

from collections import defaultdict
from trees import build_edge_map, build_index_map, print_edges
import numpy as np
import torch


class Graph:
    def __init__(self, digit_labels, tree_labels):
        self.edge_map = build_edge_map()
        self.adj_lists = defaultdict(list)
        vertices = [i for i, l in enumerate(digit_labels) if l == 1]
        for v in vertices:
            for w in vertices:
                if v != w:
                    self.adj_lists[v].append(w)
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
        priorities[v] = [np.inf, None]
    if len(graph.adj_lists) == 0:
        return tree
    min_vertex = min(v for v in graph.adj_lists)
    priorities[min_vertex] = [0, None]
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
    g = Graph(torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), torch.Tensor([0.9973, 0.9964, 0.0022, 0.0019, 0.0017, 0.0018, 0.0018, 0.0017, 0.0020,
                                                                          0.0013, 0.9930, 0.9938, 0.0016, 0.0016, 0.0017, 0.0019, 0.0016, 0.0014,
                                                                          0.0017, 0.9908, 0.9906, 0.0017, 0.0013, 0.0016, 0.0020, 0.0014, 0.0018,
                                                                          0.9822, 0.9664, 0.0019, 0.0014, 0.0021, 0.0017, 0.0019, 0.1247, 0.0013,
                                                                          0.0027, 0.0015, 0.0017, 0.0017, 0.0019, 0.0017, 0.0020, 0.0024, 0.0018]))
    g.print()
    tree = prims(g)
    print(tree)

    # change 4,9 if we're interested in indices for other indexes
    edge_map = build_edge_map()
    print(edge_map[(4, 9)])

    weights = torch.zeros(45)
    weights[edge_map[(3, 4)]] = 1
    weights[edge_map[(3, 5)]] = 1
    weights[edge_map[(4, 6)]] = 1
    weights[edge_map[(6, 7)]] = 1
    weights[edge_map[(5, 8)]] = 1
    g = Graph(torch.Tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 0]), weights)
    g.print()

    tree = prims(g)
    print(tree)

    index_map = build_index_map(edge_map)
    print_edges(tree, index_map)

    weights = torch.zeros(45)
    weights[edge_map[(0,5)]] = 1
    weights[edge_map[(0,6)]] = 1
    weights[edge_map[(6,8)]] = 1
    g = Graph(torch.Tensor([1, 0, 0, 0, 0, 1, 1, 0, 1, 0]), weights)
    g.print()

    tree = prims(g)
    print(tree)
    print_edges(tree, index_map)


if __name__ == '__main__':
    main()
