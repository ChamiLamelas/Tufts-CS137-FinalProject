import torchvision.datasets as datasets
import graphviz
# To install into conda on windows follow this answer
# https://stackoverflow.com/a/47043173

import os
import numpy as np
from PIL import Image
import random
import argparse
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from trees import Node

DIGIT_TEMP_DIR = "digit_tmp_dir"
IMAGE_SUFFIX = ".gv.png"
TREE_LABEL_SUFFIX = "_tree_label.npy"
DIGIT_LABELS_SUFFIX = "_digit_labels.npy"
DATANAME_PREFIX = "tree_"


def get_cmdline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory where tree pngs are saved')
    parser.add_argument('-n', '--num_images', type=int,
                        help='number of tree pngs to generate')
    parser.add_argument('-r', '--random_seed', type=int,
                        help='random seed so trees can be reproducible')
    parser.add_argument('-s', '--split', type=bool,
                        help='whether to take digits from MNIST train or not')
    args = parser.parse_args()
    return args.output_directory, args.num_images, args.random_seed, args.split


def invert_image(image):
    return Image.fromarray(np.invert(np.asarray(image)))


class DigitPicker:
    def __init__(self, train):
        name = f'mnist_{"train" if train else "test"}'
        self.mnist_split = datasets.MNIST(os.path.join(
            '..', 'data', name), train=train, download=True, transform=None)
        label_list = np.array([e[1] for e in self.mnist_split])
        self.label_indices = list()
        for d in range(10):
            self.label_indices.append(np.where(label_list == d)[0].tolist())

    def get_image(self, digit):
        assert 0 <= digit <= 9, f'{digit} invalid'
        idx = random.choice(self.label_indices[digit])
        return invert_image(self.mnist_split[idx][0])


def build_edge_map():
    edge_map = dict()
    k = 0
    for i in range(10):
        for j in range(i + 1, 10):
            edge_map[(i, j)] = k
            k += 1
    return edge_map


def mod_img(image):
    # https://stackoverflow.com/a/50332356
    image = image.convert('RGB')
    translist = transforms.Compose([
        # https://pytorch.org/vision/stable/generated/torchvision.transforms.Grayscale.html#torchvision.transforms.Grayscale
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    tensor = torch.squeeze(translist(image))
    imgrows = tensor.size()[0]
    imgcols = tensor.size()[1]

    if not (imgrows < 512 and imgcols < 512):
        print("Bad size, returning none")
        return None

    row_diff = 512 - imgrows
    col_diff = 512 - imgcols
    left_row_pad = int(row_diff / 2)
    top_col_pad = int(col_diff / 2)
    newtensor = torch.ones((512, 512), dtype=torch.float32)
    newtensor[left_row_pad:left_row_pad + imgrows,
              top_col_pad:top_col_pad + imgcols] = tensor
    imgconverter = transforms.ToPILImage()
    return imgconverter(newtensor)


def draw_nodes(graph, node, edge_map, digit_labels, tree_labels):
    shapes = ["ellipse", "circle", "box", "none"]
    shape_weights = [0.4, 0.2, 0.2, 0.2]

    pen_widths = ["1", "2", "0.5"]
    pen_width_weights = [0.5, 0.25, 0.25]

    heights = ["0.5", "0.25"]
    height_weights = [0.5, 0.5]

    widths = ["0.5", "0.25"]
    width_weights = [0.5, 0.5]

    arrow_shapes = ["normal", "none"]
    arrow_shape_weights = [0.5, 0.5]

    tree_size = 0

    if node is None:
        return tree_size
    graph.node(str(node.data), "", image=f'image{node.data}.png', shape=random.choices(shapes, shape_weights)[0], height=random.choices(heights, height_weights)[0], width=random.choices(widths, width_weights)[0])
    tree_size += 1
    digit_labels[node.data] = 1
    tree_size += draw_nodes(graph, node.left, edge_map, digit_labels, tree_labels)
    tree_size += draw_nodes(graph, node.right, edge_map, digit_labels, tree_labels)

    if not node.left is None:
        graph.edge(str(node.data), str(node.left.data), arrowhead=random.choices(arrow_shapes, arrow_shape_weights)[0], penwidth=random.choices(pen_widths, pen_width_weights)[0])
        tree_labels[edge_map[(node.data, node.left.data)]] = 1

    if not node.right is None:
        graph.edge(str(node.data), str(node.right.data), arrowhead=random.choices(arrow_shapes, arrow_shape_weights)[0], penwidth=random.choices(pen_widths, pen_width_weights)[0])
        tree_labels[edge_map[(node.data, node.right.data)]] = 1

    return tree_size

def tree_to_graphviz(root_node, img_name, picker, output_directory, edge_map):
    for n in range(10):
        im = picker.get_image(n)
        im.save(os.path.join(DIGIT_TEMP_DIR, f'image{n}.png'))

    dot = graphviz.Digraph(img_name, graph_attr={'imagepath': os.path.join(os.getcwd(), DIGIT_TEMP_DIR)})

    digit_labels = np.zeros(10, dtype=np.int32)
    tree_labels = np.zeros(45, dtype=np.int32)

    tree_size = draw_nodes(dot, root_node, edge_map, digit_labels, tree_labels)
    size_dist[tree_size] += 1

    dot.render(directory=output_directory, format='png')
    np.save(os.path.join(output_directory,
                         f'{img_name}{TREE_LABEL_SUFFIX}'), tree_labels)
    np.save(os.path.join(output_directory,
                         f'{img_name}{DIGIT_LABELS_SUFFIX}'), digit_labels)

    for n in range(10):
        os.remove(os.path.join(DIGIT_TEMP_DIR, f'image{n}.png'))
    img_loc = os.path.join(output_directory, f'{img_name}{IMAGE_SUFFIX}')
    modded_img = mod_img(Image.open(img_loc))
    if modded_img is not None:
        modded_img.save(img_loc)
        return True

    return dot


def gen_tree_variety():
    root_node = Node(rand_val(0))
    prob_gen_children(root_node, root_node.data + 1, 0.95)

    return root_node

# Return N such that min_val <= N <= 9
def rand_val(min_val):
    val = np.random.normal(loc=0.0, scale=5.0)
    val = int(max(min_val, min(abs(val), 9)))
    return val


def prob_gen_children(node, min_val, child_chance):
    if min_val >= 9:
        return min_val
    if random.random() < child_chance:
        left_data = rand_val(min_val)
        node.left = Node(left_data)
        min_val = left_data + 1
    if min_val >= 9:
        return min_val
    if random.random() < child_chance:
        right_data = rand_val(min_val)
        node.right = Node(right_data)
        min_val = right_data + 1
    if min_val >= 9:
        return min_val

    if not node.left is None:
        min_val = prob_gen_children(node.left, min_val, child_chance * 0.5)
    if not node.right is None:
        min_val = prob_gen_children(node.right, min_val, child_chance * 0.5)

    return min_val


def gen_tree(num_nodes, img_name, picker, output_directory, edge_map):
    shapes = ["ellipse", "circle", "box", "none"]
    shape_weights = [0.4, 0.2, 0.2, 0.2]

    pen_widths = ["1",  "2", "0.5"]
    pen_width_weights = [0.5, 0.25, 0.25]

    heights = ["0.5", "0.25"]
    height_weights = [0.5, 0.5]

    widths = ["0.5", "0.25"]
    width_weights = [0.5, 0.5]

    arrow_shapes = ["normal", "none"]
    arrow_shape_weights = [0.5, 0.5]

    for n in range(num_nodes):
        im = picker.get_image(n)
        im.save(os.path.join(DIGIT_TEMP_DIR, f'image{n}.png'))
    dot = graphviz.Digraph(img_name, graph_attr={
                           'imagepath': os.path.join(os.getcwd(), DIGIT_TEMP_DIR)})
    digit_labels = np.zeros(10, dtype=np.int32)
    tree_labels = np.zeros(45, dtype=np.int32)
    for n in range(num_nodes):
        dot.node(str(n), "", image=f'image{n}.png', shape=random.choices(shapes, shape_weights)[0], height=random.choices(heights, height_weights)[0], width=random.choices(widths, width_weights)[0])
        digit_labels[n] = 1
    for n in range(num_nodes):
        left = 2*n + 1
        right = 2*n + 2
        if left < num_nodes:
            dot.edge(str(n), str(left), arrowhead=random.choices(arrow_shapes, arrow_shape_weights)[0], penwidth=random.choices(pen_widths, pen_width_weights)[0])
            tree_labels[edge_map[(n, left)]] = 1
        if right < num_nodes:
            dot.edge(str(n), str(right), arrowhead=random.choices(arrow_shapes, arrow_shape_weights)[0], penwidth=random.choices(pen_widths, pen_width_weights)[0])
            tree_labels[edge_map[(n, right)]] = 1
    dot.render(directory=output_directory, format='png')
    np.save(os.path.join(output_directory,
            f'{img_name}{TREE_LABEL_SUFFIX}'), tree_labels)
    np.save(os.path.join(output_directory,
            f'{img_name}{DIGIT_LABELS_SUFFIX}'), digit_labels)
    for n in range(num_nodes):
        os.remove(os.path.join(DIGIT_TEMP_DIR, f'image{n}.png'))
    img_loc = os.path.join(output_directory, f'{img_name}{IMAGE_SUFFIX}')
    modded_img = mod_img(Image.open(img_loc))
    if modded_img is not None:
        modded_img.save(img_loc)
        return True
    return False


def gen_trees(output_directory, num_images, seed, split):
    picker = DigitPicker(split)
    random.seed(seed)
    try:
        os.mkdir(DIGIT_TEMP_DIR)
    except FileExistsError:
        pass
    edge_map = build_edge_map()
    i = 0
    while(i < num_images):
        num_nodes = random.randint(1, 10)
        if gen_tree(num_nodes, DATANAME_PREFIX + str(i), picker,
                 os.path.join('..', 'data', output_directory), edge_map):
            i += 1
    os.rmdir(DIGIT_TEMP_DIR)



size_dist = [0] * 11
def gen_trees_variety(output_directory, num_images, seed, split):
    picker = DigitPicker(split)
    random.seed(seed)
    try:
        os.mkdir(DIGIT_TEMP_DIR)
    except FileExistsError:
        pass
    edge_map = build_edge_map()
    i = 0
    while i < num_images:
        if tree_to_graphviz(gen_tree_variety(), DATANAME_PREFIX + str(i), picker, os.path.join('..', 'data', output_directory), edge_map):
            i += 1
    os.rmdir(DIGIT_TEMP_DIR)



def main():
    output_directory, num_images, seed, split = get_cmdline_args()
    # gen_trees(output_directory, num_images, seed, split)
    gen_trees_variety(output_directory, num_images, seed, split)

    print(size_dist)
    plt.stairs(size_dist, fill=True)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == '__main__':
    main()
