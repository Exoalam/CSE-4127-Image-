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

corrupted_image = np.log1p(corrupted_image)

ft = np.fft.fft2(corrupted_image)

shift_ft = np.fft.fftshift(ft)

mag_ft = np.abs(shift_ft)
angle = np.angle(shift_ft)
D0 = 2
c = 1
dL = 1.2
dH = 1.9
H = np.zeros((img.shape),dtype=np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        Du = (i-img.shape[0]//2)**2
        Dv = (j-img.shape[1]//2)**2
        D = np.sqrt(Du+Dv)
        t1 = np.exp(-c*D**2/D0**2)
        t1 = (dH-dL)*(1-t1)+dL
        H[i,j] = t1


mag_ft *= H

result = np.multiply(mag_ft, np.exp(1j*angle))

output = np.fft.ifftshift(result)

output = np.real(np.fft.ifft2(output))

output = np.exp(output)-1

cv2.imshow("Output", output)
cv2.waitKey(0)