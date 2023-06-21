import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

points = []

img = cv2.imread("period_input.jpg",0)



ft = np.fft.fft2(img)
shift_ft = np.fft.fftshift(ft)

mag_ft = np.abs(shift_ft)

angle = np.angle(shift_ft)
D0 = 5
def onClick(event):
    ax = event.inaxes

    if ax is not None:
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        points.append((x,y))

im = plt.imshow(1+np.log1p(mag_ft), cmap="gray")
im.figure.canvas.mpl_connect('button_press_event', onClick)
plt.show(block=True)

print(points)
ideal = np.zeros((img.shape[0], img.shape[1]))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        u = (i-points[0][1])**2
        v = (j-points[0][0])**2
        D = np.sqrt(u+v)
        if D <= D0:
            ideal[i,j] = 0  
            ideal[img.shape[0]-i,img.shape[1]-j] = 0   
        else:
            ideal[i,j] = 1    


mag_ft = mag_ft * ideal

result = np.multiply(mag_ft, np.exp(1j*angle))

result = np.fft.ifftshift(result)
result = np.real(np.fft.ifft2(result))

ideal = cv2.normalize(ideal, None, 0 , 1, cv2.NORM_MINMAX)
cv2.imshow("output",ideal)

cv2.waitKey(0)