import cv2
import numpy as np

img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

image_h = img.shape[0]
image_w = img.shape[1]

kernel = np.array(([1,2,1],[2,4,2],[1,2,1]))
kernel = kernel/16
padding = (kernel.shape[0] - 1)//2
print(img)
img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

output_image_h = image_h + kernel.shape[0] - 1
output_image_w = image_w + kernel.shape[0] - 1

output = np.zeros((output_image_h, output_image_w))

for x in range(padding, output_image_h-padding):
    for y in range(padding, output_image_w-padding):
        temp = 0
        normalize = 0
        for i in range(-padding, padding+1):
            for j in range(-padding, padding+1):
                temp += kernel[i+padding][j+padding] * img[x-i][y-j]
        #         normalize += kernel[i+padding, j+padding]
        #print(temp/255)
        output[x,y] = temp/255
        #output[x,y] /= 255 
cv2.imshow('frame', kernel)

cv2.waitKey()

cv2.destroyAllWindows()