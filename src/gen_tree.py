import graphviz
import random
import os 

indices = [1, 3, 5, 7, 2, 0, 13, 15, 17, 4]

def gen_tree(num_nodes, img_name, output_directory="tree_images"):
    dot = graphviz.Digraph(img_name, graph_attr = {'imagepath': '../data/sample_mnist'})

    for n in range(0, num_nodes):
        dot.node(str(n), str(n), image=f'image{n}.png')

    for n in range(0, num_nodes):
        if 2 * n + 1 <= num_nodes:
            dot.edge(str(n), str(2 * n + 1))
        if 2 * n + 2 <= num_nodes:
            dot.edge(str(n), str(2 * n + 2))

    dot.render(directory=output_directory, format='png')

def main():
    for i in range(10):
        gen_tree(random.randint(1, 10), "tree_" + str(i))


if __name__ == '__main__':
    main()

