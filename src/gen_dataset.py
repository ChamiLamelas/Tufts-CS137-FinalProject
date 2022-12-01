import torchvision.datasets as datasets
import graphviz
import os
import numpy as np
from PIL import Image
import random
import argparse

DIGIT_TEMP_DIR = "digit_tmp_dir"


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
        assert 0 <= digit <= 9
        idx = random.choice(self.label_indices[digit])
        return invert_image(self.mnist_split[idx][0])


def gen_tree(num_nodes, img_name, picker, output_directory):
    for n in range(num_nodes):
        im = picker.get_image(n)
        im.save(os.path.join(DIGIT_TEMP_DIR, f'image{n}.png'))
    dot = graphviz.Digraph(img_name, graph_attr={
                           'imagepath': os.path.join(os.getcwd(), DIGIT_TEMP_DIR)})
    for n in range(num_nodes):
        dot.node(str(n), "", image=f'image{n}.png')
    for n in range(num_nodes):
        if 2 * n + 1 < num_nodes:
            dot.edge(str(n), str(2 * n + 1))
        if 2 * n + 2 < num_nodes:
            dot.edge(str(n), str(2 * n + 2))
    dot.render(directory=output_directory, format='png')
    for n in range(num_nodes):
        os.remove(os.path.join(DIGIT_TEMP_DIR, f'image{n}.png'))


def gen_trees(output_directory, num_images, seed, split):
    picker = DigitPicker(split)
    random.seed(seed)
    os.mkdir(DIGIT_TEMP_DIR)
    for i in range(num_images):
        num_nodes = random.randint(1, 11)
        gen_tree(num_nodes, "tree_" + str(i), picker,
                 os.path.join('..', 'data', output_directory))
    os.rmdir(DIGIT_TEMP_DIR)


def main():
    output_directory, num_images, seed, split = get_cmdline_args()
    gen_trees(output_directory, num_images, seed, split)


if __name__ == '__main__':
    main()
