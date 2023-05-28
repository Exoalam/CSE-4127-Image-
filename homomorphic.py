import numpy as np
import matplotlib.pyplot as plt

# define the dimensions of the image
width, height = 512, 512

# create the x and y coordinate arrays
y, x = np.indices((height, width))

# define the center and width of the Gaussian
center_x, center_y = width // 2, height // 2
sigma = min(width, height) / 4

# create the 2D Gaussian
g = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

# normalize the illumination pattern to [0, 1]
g = g / g.max()

# show the illumination pattern
plt.imshow(g, cmap='gray')
plt.colorbar()
plt.title('Illumination Pattern')
plt.show()
