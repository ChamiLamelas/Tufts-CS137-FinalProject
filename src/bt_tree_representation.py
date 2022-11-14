class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

# this can be used to build a vector representation of a tree that
# can be used to rebuild a unique tree


def tree_to_vector(root):
    queue = list()
    queue.append(root)
    push_child_of_non_null = True
    vector = list()
    while len(queue) > 0 and push_child_of_non_null:
        push_child_of_non_null = False
        last_level_size = len(queue)
        for _ in range(last_level_size):
            tmp = queue.pop(0)
            if tmp is not None:
                queue.append(tmp.left)
                queue.append(tmp.right)
                push_child_of_non_null = True
            else:
                queue.append(None)
                queue.append(None)
            vector.append(tmp.data if tmp is not None else 10)
    return vector


def test_tree_to_vector():
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    assert tree_to_vector(root) == [1, 2, 3, 10, 10, 10, 10]
    root = Node(1)
    root.left = Node(2)
    root.left.right = Node(3)
    assert tree_to_vector(root) == [1, 2, 10, 10,
                                    3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    root = Node(1)
    root.right = Node(2)
    root.right.left = Node(3)
    root.right.right = Node(4)
    assert tree_to_vector(root) == [1, 10, 2, 10,
                                    10, 3, 4, 10, 10, 10, 10, 10, 10, 10, 10]


def vector_to_tree(vector):
    queue = list()
    root = Node(vector[0])
    queue.append((0, root))
    while len(queue) > 0:
        idx, node = queue.pop(0)
        left_idx = 2 * idx + 1
        right_idx = 2 * idx + 2
        if left_idx < len(vector):
            if vector[left_idx] != 10:
                node.left = Node(vector[left_idx])
                queue.append((left_idx, node.left))
            if vector[right_idx] != 10:
                node.right = Node(vector[right_idx])
                queue.append((right_idx, node.right))
    return root


def tree_equals(root1, root2):
    if root1 is None and root2 is None:
        return True
    elif (root1 is None and root2 is not None) or (root1 is not None and root2 is None):
        return False
    return root1.data == root2.data and tree_equals(root1.left, root2.left) and tree_equals(root1.right, root2.right)


def test_vector_to_tree():
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    assert tree_equals(vector_to_tree([1, 2, 3, 10, 10, 10, 10]), root)
    root = Node(1)
    root.left = Node(2)
    root.left.right = Node(3)
    assert tree_equals(vector_to_tree([1, 2, 10, 10,
                                       3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]), root)
    root = Node(1)
    root.right = Node(2)
    root.right.left = Node(3)
    root.right.right = Node(4)
    assert tree_equals(vector_to_tree([1, 10, 2, 10,
                                       10, 3, 4, 10, 10, 10, 10, 10, 10, 10, 10]), root)


def main():
    test_tree_to_vector()
    test_vector_to_tree()


if __name__ == '__main__':
    main()
