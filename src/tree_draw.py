import torchvision.datasets as datasets
from PIL import Image, ImageDraw
import numpy as np

# Build sample tree
mnist_trainset = datasets.MNIST(
    root='../data/mnist_train', train=True, download=True, transform=None)


def make_blank_im_arr():
    return np.ones((255, 255), dtype=np.uint8) * 255


def get_mnist_image(digit):
    return mnist_trainset[np.random.choice(np.where(mnist_trainset.targets == digit))]


def put_mnist(im_arr, mnist_im, top_left):
    mnist_arr = np.asarray(mnist_im)
    mnist_arr = np.abs(mnist_arr.astype(np.int32) - 255)
    im_arr[top_left[0]:top_left[0] + 28,
           top_left[1]:top_left[1] + 28] = mnist_arr


def circle_mnist(im, top_left):
    bbox = [(top_left[0] + 14 - (14 * (2 ** 0.5)), top_left[1] + 14 - (14 * (2 ** 0.5))),
            (top_left[0] + 14 + (14 * (2 ** 0.5)), top_left[1] + 14 + (14 * (2 ** 0.5)))]
    print(top_left, bbox)
    draw = ImageDraw.Draw(im)
    draw.ellipse(bbox, fill=None)


# change me
nodes = [(113, 113)]

im_arr = make_blank_im_arr()
mnist_im = mnist_trainset[0][0]
for n in nodes:
    put_mnist(im_arr, mnist_im, (113, 113))
im = Image.fromarray(im_arr)
for n in nodes:
    circle_mnist(im, (113, 113))
im.save('../data/test_drawings/sample_gentree.png')
