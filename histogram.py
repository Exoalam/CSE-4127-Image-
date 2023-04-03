import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

img_h = img.shape[0]
img_w = img.shape[1]

plt.title(label="Histogram of Input Image",fontsize=20,color="black")
plt.hist(img.ravel(),256,[0,256])
plt.show()

histogram = np.zeros(256)
for i in range(img_h):
    for j in range(img_w):
        intensity = img[i,j]
        histogram[intensity] += 1
pdf = histogram/(img_h*img_w) 
plt.title(label="PDF",fontsize=20,color="black")
plt.plot(pdf)
plt.show()

cdf = np.zeros(256)
temp = 0
for i in range(256):
    temp += pdf[i]
    cdf[i] = temp

cdf *= 255
plt.title(label="CDF",fontsize=20,color="black")
plt.plot(cdf)
plt.show()

output = np.zeros((img_h,img_w))
for i in range(img_h):
    for j in range(img_w):
        intensity = img[i,j]
        output[i,j] = np.round(255*cdf[intensity])

output = cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX)
cv2.imshow('out',output)
cv2.waitKey(0)
cv2.destroyAllWindows()   

output = output * 255

output = output.astype(np.uint8)
output_histogram = np.zeros(256)

plt.title(label="Histogram of output Image",fontsize=20,color="black")
plt.hist(output.ravel(),256,[0,256])
plt.show()

for i in range(img_h):
    for j in range(img_w):
        intensity = output[i,j]
        output_histogram[intensity] += 1
output_pdf = output_histogram/(img_h*img_w) 

plt.title(label="Output PDF",fontsize=20,color="black")
plt.plot(output_pdf)
plt.show()

output_cdf = np.zeros(256)
temp = 0
for i in range(256):
    temp += output_pdf[i]
    output_cdf[i] = temp

plt.title(label="Output CDF",fontsize=20,color="black")
plt.plot(output_cdf)
plt.show()    