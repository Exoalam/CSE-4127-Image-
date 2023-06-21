import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



img = cv2.imread('hitmiss.jpeg', 0)
r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
c_img = cv2.bitwise_not(img)
print(img.max())
cv2.imshow("input", img)

structure1 = np.array([[0,0,0],
                       [1,1,0],
                       [1,0,0]], dtype=np.uint8)
structure2 = np.array([[0,1,1],
                       [0,0,1],
                       [0,0,1]], dtype=np.uint8)
structure3 = np.array([[1,1,1],
                       [0,1,0],
                       [0,1,0]], dtype=np.uint8)
structure1 *= 255
structure2 *= 255
structure3 *= 255
w = np.ones((3,3), dtype=np.uint8)
w *= 255

structure1 = cv2.resize(structure1,(150,150),interpolation=cv2.INTER_NEAREST)
structure2 = cv2.resize(structure2,(150,150),interpolation=cv2.INTER_NEAREST)
structure3 = cv2.resize(structure3,(150,150),interpolation=cv2.INTER_NEAREST)
w = cv2.resize(w,(150,150),interpolation=cv2.INTER_NEAREST)
A1 = cv2.erode(img,structure1, iterations=1)

A2 = cv2.erode(c_img,np.subtract(w,structure1), iterations=1)

A3 = cv2.erode(img,structure2, iterations=1)
A4 = cv2.erode(c_img,np.subtract(w,structure2), iterations=1)

A5 = cv2.erode(img,structure3, iterations=1)
A6 = cv2.erode(c_img,np.subtract(w,structure3), iterations=1)
output1 = cv2.bitwise_and(A1,A2)
output2 = cv2.bitwise_and(A3,A4)
cv2.imshow("A4",output2)
output3 = cv2.bitwise_and(A5,A6)

output = cv2.bitwise_or(output1,output2)
output = cv2.bitwise_or(output,output3)
#output = cv2.normalize(output,None,0,1,cv2.NORM_MINMAX)

cv2.imshow("Output", output)

cv2.waitKey()