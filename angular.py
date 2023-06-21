import cv2
import numpy as np
import matplotlib.pyplot as plt

def angular_image(image, center, a, T):

    image = image.astype(np.float32) / 255.0

    height, width = image.shape[:2]

    y, x = np.mgrid[0:height, 0:width]

    x -= center[0]
    y -= center[1]

    distance = np.sqrt(x**2 + y**2)

    beta = np.arctan2(y,x) + a * np.sin((2*np.pi*distance)/T)
    x_new = distance * np.cos(beta)
    y_new = distance * np.sin(beta)

    x_new += center[0]
    y_new += center[1]

    # mask = distance > rmax

    # x_new[mask] = x[mask] + center[0]
    # y_new[mask] = y[mask] + center[1]

    ang_image = cv2.remap(image, x_new.astype(np.float32), y_new.astype(np.float32),
                              interpolation=cv2.INTER_LINEAR)

    return (ang_image * 255).astype(np.uint8)

image = cv2.imread("input.jpg")

center = (image.shape[1] // 2, image.shape[0] // 2)
a = 0.05
T = 50

anged_image = angular_image(image, center, a, T)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image)
ax1.set_title("Original Image")
ax1.axis("off")
ax2.imshow(anged_image)
ax2.set_title("Angular Image")
ax2.axis("off")
plt.show()
