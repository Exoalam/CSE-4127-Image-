import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def find(input, target):
    diff = input - target
    mask = np.ma.less_equal(diff, -1)
    if np.all(mask):
        c = np.abs(diff).argmin()
        return c 
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

x = 256
k = 2
u = 3

Erlang_Distribution = np.zeros(x)
for i in range(x):
    Erlang_Distribution[i] = ((i**(k-1))*math.exp(-(i/u)))/((u**k)*math.factorial(k-1))

Erlang_cdf = np.zeros(256)
temp = 0
for i in range(256):
    temp += Erlang_Distribution[i]
    Erlang_cdf[i] = temp

Erlang_cdf = cv2.normalize(Erlang_cdf, None, 0, 1, cv2.NORM_MINMAX)

Erlang_cdf *= 255
f1 = plt.figure(1)
plt.title(label="CDF",fontsize=20,color="black")
plt.plot(Erlang_cdf)    
img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

img_h = img.shape[0]
img_w = img.shape[1]

histogram = np.zeros(256)
for i in range(img_h):
    for j in range(img_w):
        intensity = img[i,j]                                               
        histogram[intensity] += 1
pdf = histogram/(img_h*img_w) 

cdf = np.zeros(256)
temp = 0
for i in range(256):
    temp += pdf[i]
    cdf[i] = temp

cdf *= 255

Erlang_cdf = np.round(Erlang_cdf)
cdf = np.round(cdf)

output_cdf = np.zeros(256)
for i in range(256):
    output_cdf[i] = find(cdf[i], Erlang_cdf)

output = np.zeros((img_h,img_w))
for i in range(img_h):
    for j in range(img_w):
        intensity = img[i,j]
        output[i,j] = np.round(output_cdf[intensity])

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

f1 = plt.figure(4)
plt.title(label="Histogram of output Image",fontsize=20,color="black")
plt.hist(output.ravel(),256,[0,256])

cv2.imshow('in',img)
cv2.imshow('out',output)
plt.show()    

cv2.waitKey(0)
cv2.destroyAllWindows()   