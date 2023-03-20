import cv2
import numpy as np
import math


img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

def gaussian(k_size, sigma):
    gas_kernel = np.zeros((k_size, k_size))
    norm = 0
    gas_padding = (gas_kernel.shape[0] - 1) // 2
    for x in range(-gas_padding, gas_padding+1):
        for y in range(-gas_padding, gas_padding+1):
            c = 1/(2*3.1416*(sigma ** 2))
            gas_kernel[x+gas_padding, y+gas_padding] = c * math.exp(-(x ** 2 + y ** 2)/(2 * sigma ** 2))
            norm += gas_kernel[x+gas_padding, y+gas_padding]
    return gas_kernel/norm

image_h = img.shape[0]
image_w = img.shape[1]
kernel_size = int(input('size: '))
sigma = float(input('sigma: '))

gaussian_kernel = gaussian(kernel_size, sigma)
padding_x = (kernel_size - 1)//2 
padding_y = (kernel_size - 1)//2 

img = cv2.copyMakeBorder(img, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT)

output_image_h = image_h + kernel_size - 1
output_image_w = image_w + kernel_size - 1

gaussian_output = np.zeros((output_image_h,output_image_w))
for x in range(padding_x, output_image_h-padding_x):
    for y in range(padding_y, output_image_w-padding_y):
        temp = 0
        for i in range(-padding_x, padding_x+1):
            for j in range(-padding_y, padding_y+1):
                temp += img[x-i, y-j]*gaussian_kernel[i+padding_x,j+padding_y]
        gaussian_output[x,y] = temp
gaussian_output = cv2.normalize(gaussian_output, None, 0, 1, cv2.NORM_MINMAX)      




cv2.imshow('input',img)
cv2.imshow('Gaussian', gaussian_output)

cv2.waitKey()
cv2.destroyAllWindows()