import cv2 
import math 
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('input.jpg',cv2.IMREAD_GRAYSCALE)

cv2.imshow('input',img)

f = np.fft.fft2(img)


shift = np.fft.fftshift(f)


mag = np.abs(shift)

angle = np.angle(shift)


f1 = plt.figure(1)
plt.imshow(np.log(np.abs(shift)),'gray')


shape = img.shape
print(shape)

x = int(input('X_axis: '))
y = int(input('Y_axis: '))
r = int(input('Distance: '))

notch=np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(math.sqrt((i-x)**2+(j-y)**2)<=r):
            notch[i,j]=0
        else:
            notch[i,j]=1
  

mag = mag*notch

f1 = plt.figure(2)
plt.imshow(np.log(np.abs(mag)),'gray')
plt.show()
        
op = np.multiply(mag,np.exp(1j*angle))


inv = np.fft.ifftshift(op)

inv = np.real(np.fft.ifft2(inv))



inv/=255

cv2.imshow("output",inv)

cv2.waitKey(0)
        


