import cv2
import numpy as np

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
image = cv2.imread('input.png')

kernel = np.array()