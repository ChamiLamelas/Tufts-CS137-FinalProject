import torchvision.datasets as datasets
from PIL import Image, ImageDraw
import numpy as np

mnist_trainset = datasets.MNIST(root='../data/mnist_train', train=True, download=True, transform=None)

mnist_im = mnist_trainset[0][0]

im_arr = np.zeros([255, 255], dtype=np.uint8)
im_arr.fill(255)
im = Image.fromarray(im_arr)
im.save('../data/test_drawings/test_white.png')

mnist_arr = np.asarray(mnist_im)
mnist_arr = np.abs(mnist_arr.astype(np.int32) - 255)
im_arr[20:48, 20:48] = mnist_arr
im = Image.fromarray(im_arr)
im.save('../data/test_drawings/test_mnist_on_white.png')

draw = ImageDraw.Draw(im)
draw.line([(0,0), (10,10)], fill=128)
im.save('../data/test_drawings/test_mnist_on_white_line.png')

draw.ellipse([(20,20), (50,50)], fill=None)
im.save('../data/test_drawings/test_mnist_on_white_circle.png')








