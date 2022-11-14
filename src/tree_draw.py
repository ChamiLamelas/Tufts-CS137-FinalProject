import torchvision.datasets as datasets
from PIL import Image, ImageDraw
import numpy as np
import random

# Build sample tree
mnist_trainset = datasets.MNIST(
    root='../data/mnist_train', train=True, download=True, transform=None)


def make_blank_im_arr():
    return np.ones((255, 255), dtype=np.uint8) * 255


def get_mnist_image(digit):
    return mnist_trainset[random.choice(np.where(mnist_trainset.targets == digit)[0])][0]


def put_mnist(im_arr, mnist_im, top_left):
    mnist_arr = np.asarray(mnist_im)
    mnist_arr = np.abs(mnist_arr.astype(np.int32) - 255)
    im_arr[top_left[0]:top_left[0] + 28,
           top_left[1]:top_left[1] + 28] = mnist_arr


def rc_to_xy(rcs):
    return [(c, r) for r, c in rcs]


def circle_mnist(top_left, drawer):
    bbox = [(top_left[0] + 14 - (14 * (2 ** 0.5)), top_left[1] + 14 - (14 * (2 ** 0.5))),
            (top_left[0] + 14 + (14 * (2 ** 0.5)), top_left[1] + 14 + (14 * (2 ** 0.5)))]
    drawer.ellipse(rc_to_xy(bbox), fill=None)


def draw_edge(drawer, start, end):
    drawer.line(rc_to_xy([start, end]), fill=0)    


nodes = [(50, 113), (100, 63), (100, 163), (150, 43)]
digits = [1, 2, 3, 4]

im_arr = make_blank_im_arr()
mnist_ims = [get_mnist_image(d) for d in digits]
for n, mnist_im in zip(nodes, mnist_ims):
    put_mnist(im_arr, mnist_im, n)
im = Image.fromarray(im_arr)
draw = ImageDraw.Draw(im)
for n in nodes:
    circle_mnist(n, draw)
draw_edge(draw, (82, 127), (93, 71))
draw_edge(draw, (82, 128), (93, 170))
draw_edge(draw, (132, 77), (144, 57))
im.save('../overleaf/images/sample_gentree.png')
