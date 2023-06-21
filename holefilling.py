import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

point_list=[]

img = cv2.imread('img2.jpg', 0)
r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
c_img = cv2.bitwise_not(img)
kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(50,50))
kernel1 = (kernel1) *255
kernel = np.uint8(kernel1)
# click and seed point set up
x = None
y = None

# The mouse coordinate system and the Matplotlib coordinate system are different, handle that
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


X = np.zeros_like(img)
plt.title("Please select seed pixel from the input")
im = plt.imshow(img, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)

print(point_list)
input = img
cv2.imshow("asda",c_img)
output1 = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
for i in point_list:
    
    while 1:
        output1[i[1],i[0]] = 1*255
        cv2.imshow("asdad",output1)
        output = cv2.dilate(output1,kernel1,iterations = 1)

        output = cv2.bitwise_and(c_img, output)
        cv2.imshow("asd",output)
        cv2.waitKey(0)
        if (output==output1).all():
            break
        output1 = output

    # for x in range(output.shape[0]):
    #     for y in range(output.shape[1]):
    #         if(output[x,y]==255):
    #             output = cv2.dilate(output,kernel1,iterations = 1)
    #             output = cv2.bitwise_and(c_img, output)
    



output = cv2.bitwise_or(output1, input)
rate = 50
cv2.imshow("input", img)
cv2.imshow("kernel",kernel1)
cv2.imshow("Dilation", output)
cv2.waitKey()


