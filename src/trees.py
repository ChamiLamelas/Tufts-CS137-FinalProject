from heapq import heapify
from copy import deepcopy

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


def count(root):
    return 0 if root is None else 1 + count(root.left) + count(root.right)

# builds dict (edge tuple) -> int index
def build_edge_map(max_num_nodes=10):
    edge_map = dict()
    k = 0
    for i in range(max_num_nodes):
        for j in range(i + 1, max_num_nodes):
            edge_map[(i, j)] = k
            k += 1
    return edge_map

# builds dict int index -> (edge tuple)
def build_index_map(edge_map):
    return {v: k for k, v in edge_map.items()}

# builds tree from vector representing edge set
# 1 Numbers are unique in 0...max_num_nodes
# 2 Edges go low to high
# 3 Going L-R over tree_vector, we add left before right
def build_tree_from_vector(tree_vector, index_map, max_num_nodes):
    nodes = [Node(i) for i in range(max_num_nodes)]
    found_root = False
    for i, edge_exists in enumerate(tree_vector):
        if edge_exists == 1:
            edge = index_map[i]
            if not found_root:
                root = nodes[edge[0]]
                found_root = True
            if nodes[edge[0]].left is None:
                nodes[edge[0]].left = nodes[edge[1]]
            else:
                nodes[edge[0]].right = nodes[edge[1]]
    return root

# builds complete tree from level order traversal
def build_tree_level_order(values):
    root = Node(values[0])
    q = [root]
    i = 1
    while i < len(values):
        front = q.pop(0)
        front.left = Node(values[i])
        q.append(front.left)
        i += 1
        if i < len(values):
            front.right = Node(values[i])
            q.append(front.right)
            i += 1
    return root

# given what will be level order traversal of node values, ensures each level is sorted low to high
# satisfies req 3 of build_tree_from_vector
def level_sort(node_values):
    i = 0
    level_size = 1
    out = list()
    while i < len(node_values):
        ext = sorted(node_values[i:i+level_size])
        out.extend(ext)
        i += level_size
        level_size *= 2
    return out

# given what will be a level order traversal of node values, makes sure they will build an allowed
# tree parents are smaller than their children and that L->R across levels is increasing
def turn_into_valid_level_order(node_values):
    cpy = deepcopy(node_values)
    heapify(cpy)
    return level_sort(cpy)

def main():
    edge_map = build_edge_map(4)
    index_map = build_index_map(edge_map)

    root = build_tree_from_vector([1,0,0,1,0,1], index_map, 4)
    print(root.data, root.left.data, root.left.left.data, root.left.left.left.data, count(root))

    root = build_tree_from_vector([1,0,1,1,0,0], index_map, 4)
    print(root.data, root.left.data, root.right.data, root.left.left.data, count(root))

    root = build_tree_level_order([1,3,4,6])
    print(root.data, root.left.data, root.right.data, root.left.left.data, count(root))

    values = [6,5,3,2]
    print(turn_into_valid_level_order(values))

if __name__ == '__main__':
    main()
