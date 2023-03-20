import cv2
import numpy as np

image = cv2.imread('contrast.tif',cv2.IMREAD_GRAYSCALE)

output = np.zeros((image.shape))

img_max = image.max(axis=(0,1))
img_min = image.min(axis=(0,1))
for x in range(0, image.shape[0]):
    for y in range(0, image.shape[1]):
        output[x,y] = ((image[x,y] - img_min)/(img_max - img_min))*255

output = cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.imshow('Input', image)
cv2.imshow('Contrast Output',output)       

cv2.waitKey()
cv2.destroyAllWindows()