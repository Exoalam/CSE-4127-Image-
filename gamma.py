import cv2
import numpy as np

image = cv2.imread('input.png',cv2.IMREAD_GRAYSCALE)

output = np.zeros((image.shape))
gamma = 2.2
c = 1
for x in range(0,image.shape[0]):
    for y in range(0,image.shape[1]):
        output[x,y] = c * (image[x,y] ** gamma)  

output = cv2.normalize(output,None,0,1,cv2.NORM_MINMAX)

print(output)

cv2.imshow('Input', image)
cv2.imshow('Gamma Output',output)       

cv2.waitKey()
cv2.destroyAllWindows()
