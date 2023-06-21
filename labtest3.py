import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("input.png", 0)

f1 = plt.figure(1)
plt.title(label="Input Histogram")
plt.hist(img.ravel(),255,[0,255])

input_pdf = np.zeros(256)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        intensity = img[i,j]
        input_pdf[intensity] += 1

input_pdf = input_pdf/(img.shape[0]*img.shape[1])
f1 = plt.figure(2)
plt.title(label="Input PDF")
plt.plot(input_pdf)

input_cdf = np.zeros(256)
temp = 0
for i in range(256):
    temp += input_pdf[i]
    input_cdf[i] = temp

input_cdf *= 255

f1 = plt.figure(3)
plt.title(label="Input CDF")
plt.plot(input_cdf)


output = np.zeros((img.shape))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        intensity = img[i,j]
        output[i,j] = np.round(input_cdf[intensity])

output = output * 255

output = output.astype(np.uint8)

output_pdf = np.zeros(256)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        intensity = output[i,j]
        output_pdf[intensity] += 1

output_pdf = output_pdf/(img.shape[0]*img.shape[1])
f1 = plt.figure(5)
plt.title(label="Output PDF")
plt.plot(output_pdf)

output_cdf = np.zeros(256)
temp = 0
for i in range(256):
    temp += output_pdf[i]
    output_cdf[i] = temp

output_cdf *= 255

f1 = plt.figure(6)
plt.title(label="Output CDF")
plt.plot(output_cdf)
f1 = plt.figure(4)
plt.title(label="Output")
plt.hist(output.ravel(), 255,[0,255])
plt.show()