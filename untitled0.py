import cv2
import numpy as np


img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
height, width = 512, 512
angle = np.deg2rad(45)
x = np.linspace(-1, 1, width)
y = np.linspace(-1, 1, height)

xx, yy = np.meshgrid(x, y)

x_angle = np.cos(angle)
y_angle = np.sin(angle)

illumination = xx * x_angle + yy * y_angle

illumination -= illumination.min()

illumination /= illumination.max()

corrupted_image = np.multiply(img, illumination)

corrupted_image = cv2.normalize(corrupted_image, None, 0, 1, cv2.NORM_MINMAX)



ft = np.fft.fft2(corrupted_image)

shit_ft = np.fft.fftshift(ft)

