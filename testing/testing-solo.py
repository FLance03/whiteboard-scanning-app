import testing

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def onclick(e):
    print('{x}, {y}'.format(x=int(e.xdata), y=int(e.ydata)), end=', ')
def onrelease(e):
    print('{x}, {y}'.format(x=int(e.xdata), y=int(e.ydata)))

# img = cv.imread('11.png')
data = np.load('output.npz')['data']
img = data[8]
# retImg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
# for r, row in enumerate(img):
#     for c, col in enumerate(row):
#         if np.all(col == [0, 255, 135]):
#             retImg[max(0, r-5):min(len(retImg), r+5), max(0, c-5):min(len(retImg[0]), c+5)] = 255
# img = retImg

# plt.subplot(1, 1, 1)
# fig, ax = plt.subplots()
plt.imshow(img, 'gray') if len(img.shape) == 2 else plt.imshow(img)
# plt.title('Image')
# fig.canvas.mpl_connect('button_press_event', onclick)
# fig.canvas.mpl_connect('button_release_event', onrelease)
plt.show()