import graphviz
import random

def gen_tree(num_nodes, img_name, output_directory="tree_images"):
    dot = graphviz.Digraph(img_name)

    for n in range(1, num_nodes):
        dot.node(str(n), str(n))

    for n in range(1, num_nodes):
        if 2 * n <= num_nodes:
            dot.edge(str(n), str(2 * n))
        if 2 * n + 1 <= num_nodes:
            dot.edge(str(n), str(2 * n + 1))

    dot.render(directory=output_directory, format='png')

def main():
    for i in range(10):
        gen_tree(random.randint(2, 11), "tree_" + str(i))


if __name__ == '__main__':
    main()

