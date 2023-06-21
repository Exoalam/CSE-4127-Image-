import cv2
import numpy as np





def gaussian(n,s):
    x = (n-1)//2
    kernel = np.zeros((n,n))
    sum = 0
    for i in range(-x,x+1):
        for j in range(-x,x+1):
            kernel[i+x,j+x] = np.exp(-(i**2+j**2)/(2*s**2))
            sum += kernel[i+x,j+x]
    return kernel/sum

n = int(input("Size: "))
s = int(input("Sigma: "))

kernel = gaussian(n,s)

img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
image_size = img.shape
padding = (n-1)//2
sigma = 1.4
img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
image_size = image_size[0] + n - 1
output = np.zeros((image_size,image_size))
for x in range(padding,img.shape[0]-padding):
    for y in range(padding,img.shape[1]-padding):
        temp = 0
        sum = 0
        out = 0
        p = img[x,y]
        for i in range(-padding,padding+1):
            for j in range(-padding,padding+1):
                q = img[x-i,y-j]
                temp = kernel[i+padding,j+padding]*np.exp(-(np.subtract(p,q)**2)/(2*sigma**2))
                sum += temp
                out += temp * q
        output[x,y] = out/sum

output = cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX)

cv2.imshow("output", output)
cv2.waitKey(0)
