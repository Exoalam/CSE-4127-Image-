import cv2
import numpy as np
import matplotlib.pyplot as plt

def twirl_image(image, center, angle, rmax):
    # Convert image to floating point precision
    image = image.astype(np.float32) / 255.0

    # Calculate the dimensions of the image
    height, width = image.shape[:2]

    # Create a grid of coordinates
    y, x = np.mgrid[0:height, 0:width]

    # Translate coordinates to center
    x -= center[0]
    y -= center[1]

    # Calculate the distance from the center
    distance = np.sqrt(x**2 + y**2)

    # Calculate the increasing angle based on distance
    angle_increase = angle * (distance / rmax)

    # Calculate the new coordinates after twirl operation
    x_new = x * np.cos(angle_increase) - y * np.sin(angle_increase)
    y_new = x * np.sin(angle_increase) + y * np.cos(angle_increase)

    # Translate coordinates back to original position
    x_new += center[0]
    y_new += center[1]

    # Create a mask for pixels outside the radial distance
    mask = distance > rmax

    # Set the original pixel values for the pixels outside the radial distance
    x_new[mask] = x[mask] + center[0]
    y_new[mask] = y[mask] + center[1]

    # Interpolate the pixel values using the new coordinates
    twirled_image = cv2.remap(image, x_new.astype(np.float32), y_new.astype(np.float32),
                              interpolation=cv2.INTER_LINEAR)

    return (twirled_image * 255).astype(np.uint8)

# Load the image
image = cv2.imread("input.png")

# Define the center point and parameters for the twirl operation
center = (image.shape[1] // 2, image.shape[0] // 2)
rotation_angle = 45  # Angle of rotation (in degrees)
max_radius = min(center[0], center[1])  # Maximum radial distance (rmax)

# Apply the twirl operation to the image
twirled_image = twirl_image(image, center, np.radians(rotation_angle), max_radius)

# Convert the image from BGR to RGB (for displaying with matplotlib)
twirled_image_rgb = cv2.cvtColor(twirled_image, cv2.COLOR_BGR2RGB)

# Display the original and twirled images using matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title("Original Image")
ax1.axis("off")
ax2.imshow(twirled_image_rgb)
ax2.set_title("Twirled Image")
ax2.axis("off")
plt.show()
