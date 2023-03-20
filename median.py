import cv2
import numpy as np
import math


img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

image_h = img.shape[0]
image_w = img.shape[1]
kernel_size = int(input('size: '))

padding_x = (kernel_size - 1)//2 
padding_y = (kernel_size - 1)//2 

img = cv2.copyMakeBorder(img, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT)

output_image_h = image_h + kernel_size - 1
output_image_w = image_w + kernel_size - 1

median_output = np.zeros((output_image_h,output_image_w))
median = (kernel_size*kernel_size)//2
for x in range(padding_x, output_image_h-padding_x):
    for y in range(padding_y, output_image_w-padding_y):
        temp = []
        for i in range(-padding_x, padding_x+1):
            for j in range(-padding_y, padding_y+1):
                temp.append(img[x-i,y-j])
        temp.sort()
        median_output[x,y] = temp[median]
median_output = cv2.normalize(median_output, None, 0, 1, cv2.NORM_MINMAX)      



cv2.imshow('input',img)
cv2.imshow('median', median_output)

cv2.waitKey()
cv2.destroyAllWindows()