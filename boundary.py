import numpy as np
import cv2
import matplotlib.pyplot as plt

def erode(image, kernel):
    height, width = image.shape
    kernel_size = kernel.shape[0]
    kernel_offset = kernel_size // 2
    result = np.zeros_like(image)

    for i in range(kernel_offset, height - kernel_offset):
        for j in range(kernel_offset, width - kernel_offset):
            if np.all(image[i - kernel_offset:i + kernel_offset + 1, j - kernel_offset:j + kernel_offset + 1] == kernel):
                result[i, j] = 1

    return result


# Example usage
image = cv2.imread("j.png", cv2.IMREAD_GRAYSCALE)
image = np.divide(image, 255)
kernel = np.ones((5,5),np.uint8)
eroded_image = erode(image, kernel)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i,j] == eroded_image[i,j]:
            eroded_image[i,j] = 1
        else:
            eroded_image[i,j] = 0    
cv2.imshow("output", eroded_image)
cv2.waitKey(0)
