import cv2 #opencv itself
import numpy as np # matrix manipulations

from matplotlib import pyplot as plt # this lets you draw inline pictures in the notebooks
import pylab # this allows you to control figure size
pylab.rcParams['figure.figsize'] = (10.0, 8.0) # this controls figure size in the notebook

input_image=cv2.imread('/Users/wanghe/Desktop/test.png')
#bgr2rgb
flipped_code_0=cv2.flip(input_image,5)
rgb_image = cv2.cvtColor(flipped_code_0, cv2.COLOR_BGR2RGB)
# vertical flip
plt.imshow(rgb_image)
plt.show()