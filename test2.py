import numpy as np
import cv2

def adaptive_median_filter(img, S_max):
    new_img = img.copy()
    n = 1
    while n <= S_max:
        z_min, z_max, z_med = get_z_values(img, n)
        A1 = z_med - z_min
        A2 = z_med - z_max
        if A1 > 0 and A2 < 0:
            B1 = img - z_min
            B2 = img - z_max
            if B1 > 0 and B2 < 0:
                new_img = img
            else:
                new_img = z_med
            break
        else:
            n += 1
    return new_img

def get_z_values(img, n):
    size = 2*n + 1
    kernel = np.ones((size, size))
    local_area = cv2.filter2D(img, -1, kernel)
    z_min = np.min(local_area)
    z_max = np.max(local_area)
    z_med = np.median(local_area)
    return z_min, z_max, z_med

# Load the image
img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

# Apply the filter
filtered_img = adaptive_median_filter(img, S_max=5)

# Save the result
cv2.imwrite('filtered_image.png', filtered_img)
