import numpy as np
from matplotlib import pyplot as plt
import skimage.io
from skimage.transform import rescale, resize
from mpl_toolkits.mplot3d import Axes3D
import png

def save_depth(path, im, width, height):
    """Saves a depth image (16-bit) to a PNG file.
    From the BOP toolkit (https://github.com/thodan/bop_toolkit).

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if not path.endswith(".png"):
        raise ValueError('Only PNG format is currently supported.')

    im[im > 65535] = 65535
    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(width, height, greyscale=True, bitdepth=16)
    with open(path, 'wb') as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, width))) 

mat = np.load("3D/Photo1/mat.npy")

img = skimage.io.imread('3D/Photo1/1_Color.png')

mat = mat.transpose()

plt.imshow(mat, interpolation='none')
plt.imsave("3D/Photo1/1_depth.png", mat)
plt.show()