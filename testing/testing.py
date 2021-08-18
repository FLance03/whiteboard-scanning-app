from pprint import pprint as p

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

dir = 'testing/pics and texts/'

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

def WriteFile(someObject, pretty=False):
    with open(dir + 'print2.txt', 'w') as out:
        print(someObject, file=out) if not pretty else p(someObject, stream=out)

def FullPrint(*args, **kwargs):
  opt = np.get_printoptions()
  np.set_printoptions(threshold=np.inf)
  with open(dir + 'print.txt', 'w') as out:
    print(*args, **kwargs, file=out)
  np.set_printoptions(**opt)

def FullPrint2(filename, *args, **kwargs):
  opt = np.get_printoptions()
  np.set_printoptions(threshold=np.inf)
  with open(dir + filename + '.txt', 'w') as out:
    print(*args, **kwargs, file=out)
  np.set_printoptions(**opt)

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img
def PlotIt(img):
    plt.subplot(1, 1, 1)
    plt.imshow(img, 'gray') if len(img.shape) == 2 else plt.imshow(img)
    plt.title('Image')
    plt.show()