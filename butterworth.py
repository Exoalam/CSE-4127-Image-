import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')



img = cv2.imread("period_input.jpg", cv2.IMREAD_GRAYSCALE)

ft = np.fft.fft2(img)
ft = np.fft.fftshift(ft)
mag_ft = np.abs(ft)
f_input = 1 * np.log(np.abs(ft)+1)
m = img.shape[0]//2
n = img.shape[1]//2
point_list=[]
def onclick(event):
    global x, y
    ax = event.inaxes
    if ax is not None:

        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, x, y))
        point_list.append((x,y))


plt.title("Please select seed pixel from the input")
im = plt.imshow(f_input, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)
print(point_list)
D0 = 15
N = 1
bw=np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        Du = (i-m-point_list[0][1])**2
        Dv = (j-n-point_list[0][0])**2
        Dk = np.sqrt(Du+Dv)
        _Du = (i-m+point_list[0][1])**2
        _Dv = (j-n+point_list[0][0])**2
        _Dk = np.sqrt(_Du+_Dv)
        DP1 = 1+(D0/Dk)**(2*N)
        DP2 = 1+(D0/_Dk)**(2*N) 
        H = (1/DP1) * (1/DP2)
        bw[i,j] = H

filter = cv2.normalize(bw, None,0,1, cv2.NORM_MINMAX)
mag_ft = mag_ft * bw

ang = np.angle(ft)

result = np.multiply(mag_ft, np.exp(1j*ang))

output = np.real(np.fft.ifft2(np.fft.ifftshift(result)))

magnitude_spectrum = cv2.normalize(f_input, None,0,1, cv2.NORM_MINMAX)

output = cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX)

cv2.imshow("Input", img)
cv2.imshow("filter",filter)
cv2.imshow("Fourier", magnitude_spectrum)
cv2.imshow("Output", output)
cv2.waitKey()