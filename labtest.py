import cv2
import numpy as np

img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)

kernal = np.ones((10,10))

kernal = kernal/100

center = (3,3)

top,left = center
bottom,right = 9-3, 9-3
padding_x = (kernal.shape[0]-1)//2
padding_y = (kernal.shape[1]-1)//2

img = cv2.copyMakeBorder(img, bottom, top, right, left, cv2.BORDER_CONSTANT)

output = np.zeros((img.shape))
for i in range(right, img.shape[0]-left):
    for j in range(bottom, img.shape[1]-top):
        temp = 0
        for x in range(-left,right+1):
            for y in range(-top,bottom+1):
                temp += img[i-x,j-y]*kernal[x+left,y+top]
        output[i,j] = temp

output = cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX)

cv2.imshow("Output", output)
cv2.waitKey(0)