import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

img_h = img.shape[0]
img_w = img.shape[1]
f1 = plt.figure(1)
plt.title(label="Histogram of Input Image",fontsize=20,color="black")
plt.hist(img.ravel(),256,[0,256])

histogram = np.zeros(256)
for i in range(img_h):
    for j in range(img_w):
        intensity = img[i,j]
        histogram[intensity] += 1
pdf = histogram/(img_h*img_w) 
f1 = plt.figure(2)
plt.title(label="PDF",fontsize=20,color="black")
plt.plot(pdf)

cdf = np.zeros(256)
temp = 0
for i in range(256):
    temp += pdf[i]
    cdf[i] = temp

cdf *= 255
f1 = plt.figure(3)
plt.title(label="CDF",fontsize=20,color="black")
plt.plot(cdf)


output = np.zeros((img_h,img_w))
for i in range(img_h):
    for j in range(img_w):
        intensity = img[i,j]
        output[i,j] = np.round(cdf[intensity])

output = cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX)


output = output * 255

output = output.astype(np.uint8)
output_histogram = np.zeros(256)
f1 = plt.figure(4)
plt.title(label="Histogram of output Image",fontsize=20,color="black")
plt.hist(output.ravel(),256,[0,256])


for i in range(img_h):
    for j in range(img_w):
        intensity = output[i,j]
        output_histogram[intensity] += 1
output_pdf = output_histogram/(img_h*img_w) 
f1 = plt.figure(5)
plt.title(label="Output PDF",fontsize=20,color="black")
plt.plot(output_pdf)


output_cdf = np.zeros(256)
temp = 0
for i in range(256):
    temp += output_pdf[i]
    output_cdf[i] = temp

output_cdf *= 255
f1 = plt.figure(6)
plt.title(label="Output CDF",fontsize=20,color="black")
plt.plot(output_cdf)
cv2.imshow('in',img)
cv2.imshow('out',output)
plt.show()    

cv2.waitKey(0)
cv2.destroyAllWindows()   