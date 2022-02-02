import numpy as np
from matplotlib import pyplot as plt
import skimage.io
from skimage.transform import rescale, resize
from mpl_toolkits.mplot3d import Axes3D
import png

mat = np.load("Photo1/mat.npy")

img = skimage.io.imread('Photo1/1_Color.png')

plt.imshow(mat, interpolation='none')
#plt.imsave("3D/Photo1/1_depth.png", mat)
plt.show()