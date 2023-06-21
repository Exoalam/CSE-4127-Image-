import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("input.png")

def his(img):
    input_pdf = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            intensity = img[i,j]
            input_pdf[intensity] += 1

    input_pdf = input_pdf/(img.shape[0]*img.shape[1])

    input_cdf = np.zeros(256)
    temp = 0
    for i in range(256):
        temp += input_pdf[i]
        input_cdf[i] = temp

    input_cdf *= 255


    output = np.zeros((img.shape))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            intensity = img[i,j]
            output[i,j] = np.round(input_cdf[intensity])

    output = cv2.normalize(output,None,0,1,cv2.NORM_MINMAX)
    return output

b,g,r = cv2.split(img)



b_o = his(b)
g_o = his(g)
r_o = his(r)
out = cv2.merge((b_o,g_o,r_o))
cv2.imshow("input", img)
cv2.imshow("out",out)
cv2.waitKey(0)