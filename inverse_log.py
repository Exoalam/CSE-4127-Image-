import cv2
import numpy as np
import math

image = cv2.imread('input.png',cv2.IMREAD_GRAYSCALE)

output = np.zeros((image.shape))

max_value = image.max(axis=(0,1))
print(max_value)
c = 255/(math.log(1+max_value))
for x in range(0,image.shape[0]):
    for y in range(0,image.shape[1]):
        output[x,y] = (math.exp(image[x,y]) ** (1/c)) - 1

output = cv2.normalize(output,None,0,1,cv2.NORM_MINMAX)

print(output)

cv2.imshow('Input', image)
cv2.imshow('Inverse_log Output',output)       

cv2.waitKey()
cv2.destroyAllWindows()
