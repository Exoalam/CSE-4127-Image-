import cv2
import numpy as np
import math


img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

image_h = img.shape[0]
image_w = img.shape[1]

kernel = np.array(([1,2,1],[2,4,2],[1,2,1]))
kernel = kernel/16
padding_x = (kernel.shape[0] - 1)//2
padding_y = (kernel.shape[1] - 1)//2
img = cv2.copyMakeBorder(img, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT)

output_image_h = image_h + kernel.shape[0] - 1
output_image_w = image_w + kernel.shape[1] - 1

output = np.zeros((output_image_h, output_image_w))

gamma_output = np.zeros((image_h, image_w))

inv_log_output = np.zeros((image_h, image_w))
# Gamma
# c = 1
# gamma = 0.8
# for x in range(0, image_h):
#     for y in range(0, image_w):
#         gamma_output[x,y] = (c * (img[x,y] ** gamma)) / 255

# Inverse-Log
# c = 255/(math.log(1+img.max(axis=(0, 1))))
# for x in range(0, image_h):
#     for y in range(0, image_w):
#         inv_log_output[x,y] = (math.exp(img[x,y]) ** (1/c) - 1)/255
        #inv_log_output[x,y] = (c * math.log(1+img[x,y]))/255
#gaussian
# for x in range(padding, output_image_h-padding):
#     for y in range(padding, output_image_w-padding):
#         temp = 0
#         normalize = 0
#         for i in range(-padding, padding+1):
#             for j in range(-padding, padding+1):
#                 temp += kernel[i+padding][j+padding] * img[x-i][y-j]
#         #         normalize += kernel[i+padding, j+padding]
#         #print(temp/255)
#         output[x,y] = temp/255
#         #output[x,y] /= 255 
# mean
# for x in range(padding, output_image_h-padding):
#     for y in range(padding, output_image_w-padding):
#         temp = 0
#         for i in range(-padding, padding+1):
#             for j in range(-padding, padding+1):
#                 temp += img[x-i, y-j]
#         output[x,y] = temp/(kernel.shape[0] * kernel.shape[1])
#         output[x,y] /= 255

cv2.imshow('frame', inv_log_output)

cv2.waitKey()

cv2.destroyAllWindows()

