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

gaussina_output = np.zeros((output_image_h, output_image_w))

mean_output = np.zeros((output_image_h, output_image_w))

median_output = np.zeros((output_image_h, output_image_w))

gamma_output = np.zeros((image_h, image_w))

inv_log_output = np.zeros((image_h, image_w))

contrast_stretching_output = np.zeros((image_h, image_w))
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

# Contrast Stretching
# img_max = img.max(axis=(0,1))
# img_min = img.min(axis=(0,1))
# print(img_min, img_max)
# for x in range(0, image_h):
#     for y in range(0, image_w):
#         contrast_stretching_output[x,y] = ((img[x,y] - img_min)/(img_max - img_min))*255
# contrast_stretching_output /= 255

# mean
# for x in range(padding, output_image_h-padding):
#     for y in range(padding, output_image_w-padding):
#         temp = 0
#         for i in range(-padding, padding+1):
#             for j in range(-padding, padding+1):
#                 temp += img[x-i, y-j]
#         mean_output[x,y] = temp/(kernel.shape[0] * kernel.shape[1])
#         mean_output[x,y] /= 255

#median
# median = (kernel.shape[0]*kernel.shape[1])//2
# for x in range(padding_x, output_image_h-padding_x):
#     for y in range(padding_y, output_image_w-padding_y):
#         temp = []
#         for i in range(-padding_x, padding_x+1):
#             for j in range(-padding_y, padding_y+1):
#                 temp.append(img[x-i,y-j])
#         temp.sort()
#         median_output[x,y] = temp[median]/255

#gaussian
#function
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

cv2.imshow('input', img)
cv2.imshow('output', median_output)

cv2.waitKey()

cv2.destroyAllWindows()

