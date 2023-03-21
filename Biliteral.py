import cv2
import numpy as np
import math
def gaussian(k_size, sigma):
    gas_kernel = np.zeros((k_size, k_size),dtype=np.float32)
    norm = 0
    gas_padding = (gas_kernel.shape[0] - 1) // 2
    for x in range(-gas_padding, gas_padding+1):
        for y in range(-gas_padding, gas_padding+1):
            c = 1/(2*3.1416*(sigma ** 2))
            gas_kernel[x+gas_padding, y+gas_padding] = c * math.exp(-(x ** 2 + y ** 2)/(2 * sigma ** 2))
            norm += gas_kernel[x+gas_padding, y+gas_padding]
    return gas_kernel/norm


image = cv2.imread('onion.PNG',cv2.IMREAD_GRAYSCALE)

k_size = 3
sigma = 5
kernel = gaussian(k_size, sigma)
image_h = image.shape[0]
image_w = image.shape[1]
# kernel_size = int(input('size: '))
# sigma = float(input('sigma: '))

padding_x = (k_size - 1)//2 
padding_y = (k_size - 1)//2 

image = cv2.copyMakeBorder(image, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT)

output_image_h = image_h + k_size - 1
output_image_w = image_w + k_size - 1

biliteral_output = np.zeros((output_image_h,output_image_w),dtype=np.float32)

for x in range(padding_x, output_image_h-padding_x):
    for y in range(padding_y, output_image_w-padding_y):
        temp = 0
        div = 0
        conv = 0
        p = image[x,y]
        for i in range(-padding_x, padding_x+1):
            for j in range(-padding_y, padding_y+1):
                q = image[x-i,y-j]
                temp = kernel[i+padding_x,j+padding_y]*np.exp(-((p-q)**2)/(2*sigma**2))
                div += temp
                conv += temp * image[x-i,y-j]
                
        biliteral_output[x,y] = conv/div         
biliteral_output = cv2.normalize(biliteral_output, None, 0, 1, cv2.NORM_MINMAX)      
print(biliteral_output)
cv2.imshow('input', image)
cv2.imshow('biliteral', biliteral_output)

cv2.waitKey()
cv2.destroyAllWindows()