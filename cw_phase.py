# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dpc



def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')

# take input
img_input = cv2.imread('period_input4.jpeg', 0)

img = dpc(img_input)

image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac=np.abs(ft_shift)
magnitude_spectrum = 1 * np.log(np.abs(ft_shift)+1)
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

x = int(input('X_axis: '))
y = int(input('Y_axis: '))
_x = img.shape[0] - x
_y = img.shape[1] - y
r = int(input('Distance: '))

notch=np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(np.sqrt((i-x)**2+(j-y)**2)<=r):
            notch[i,j]=0
        else:
            notch[i,j]=1
notch2=np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(np.sqrt((i-_x)**2+(j-_y)**2)<=r):
            notch2[i,j]=0
        else:
            notch2[i,j]=1

magnitude_spectrum_ac = magnitude_spectrum_ac*notch*notch2



#mag, ang = cv2.cartToPolar(ft_shift[:,:,0],ft_shift[:,:,1])
ang = np.angle(ft_shift)

## phase add
final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_normalize(img_back)
magnitude_spectrum_scaled_x = min_max_normalize(magnitude_spectrum_scaled*notch*notch2)
## plot
cv2.imshow("input", img_input)
cv2.imshow("Magnitude Spectrum",magnitude_spectrum_scaled)
cv2.imshow("Magnitude After",magnitude_spectrum_scaled_x)
cv2.imshow("Notch", notch*notch2)
cv2.imshow("Phase",ang)
cv2.imshow("Inverse transform",img_back_scaled)



cv2.waitKey(0)
cv2.destroyAllWindows() 
