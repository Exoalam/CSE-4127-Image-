import cv2
import numpy as np
import math


img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

image_h = img.shape[0]
image_w = img.shape[1]
kernel_size = int(input('size: '))


def mean(k_size):
    kernel = np.full((k_size, k_size),1)
    kernel = kernel/(k_size*k_size)
    return kernel

mean_kernel = mean(kernel_size)
padding_x = (mean_kernel.shape[0] - 1)//2 
padding_y = (mean_kernel.shape[1] - 1)//2 

img = cv2.copyMakeBorder(img, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT)

output_image_h = image_h + mean_kernel.shape[0] - 1
output_image_w = image_w + mean_kernel.shape[1] - 1

mean_output = np.zeros((output_image_h,output_image_w))
for x in range(padding_x, output_image_h-padding_x):
    for y in range(padding_y, output_image_w-padding_y):
        temp = 0
        for i in range(-padding_x, padding_x+1):
            for j in range(-padding_y, padding_y+1):
                temp += img[x-i, y-j]*mean_kernel[i+padding_x,j+padding_y]
        mean_output[x,y] = temp
mean_output = cv2.normalize(mean_output, None, 0, 1, cv2.NORM_MINMAX)



cv2.imshow('input',img)
cv2.imshow('mean', mean_output)

cv2.waitKey()
cv2.destroyAllWindows()