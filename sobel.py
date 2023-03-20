import cv2
import numpy as np
import math


img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

image_h = img.shape[0]
image_w = img.shape[1]
kernel_size = 3

sobel_kernel_horizontal = np.array(([1,0,-1],
                                    [2,0,-2],
                                    [1,0,-1]))

sobel_kernel_vertical = np.array(([1,2,1],
                                    [0,0,0],
                                    [-1,-2,-1]))
padding_x = (kernel_size - 1)//2 
padding_y = (kernel_size - 1)//2 

img = cv2.copyMakeBorder(img, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT)

output_image_h = image_h + kernel_size - 1
output_image_w = image_w + kernel_size - 1

sobel_output_h = np.zeros((output_image_h,output_image_w))
for x in range(padding_x, output_image_h-padding_x):
    for y in range(padding_y, output_image_w-padding_y):
        temp = 0
        for i in range(-padding_x, padding_x+1):
            for j in range(-padding_y, padding_y+1):
                temp += img[x-i, y-j]*sobel_kernel_horizontal[i+padding_x,j+padding_y]
        sobel_output_h[x,y] = temp
sobel_output_h = cv2.normalize(sobel_output_h, None, 0, 1, cv2.NORM_MINMAX)

sobel_output_v = np.zeros((output_image_h,output_image_w))
for x in range(padding_x, output_image_h-padding_x):
    for y in range(padding_y, output_image_w-padding_y):
        temp = 0
        for i in range(-padding_x, padding_x+1):
            for j in range(-padding_y, padding_y+1):
                temp += img[x-i, y-j]*sobel_kernel_vertical[i+padding_x,j+padding_y]
        sobel_output_v[x,y] = temp
sobel_output_v = cv2.normalize(sobel_output_v, None, 0, 1, cv2.NORM_MINMAX)



cv2.imshow('input',img)
cv2.imshow('sobel_h', sobel_output_h)
cv2.imshow('sobel_v', sobel_output_v)
cv2.imshow('combine', sobel_output_h*sobel_output_v)
cv2.waitKey()
cv2.destroyAllWindows()