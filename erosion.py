import numpy as np

def erode(image, kernel):
    height, width = image.shape
    kernel_size = kernel.shape[0]
    kernel_offset = kernel_size // 2
    result = np.zeros_like(image)

    for i in range(kernel_offset, height - kernel_offset):
        for j in range(kernel_offset, width - kernel_offset):
            print(image[i - kernel_offset:i + kernel_offset + 1,j - kernel_offset:j + kernel_offset + 1])
            if np.all(image[i - kernel_offset:i + kernel_offset + 1, j - kernel_offset:j + kernel_offset + 1] == kernel):
                result[i, j] = 1

    return result

# Example usage
image = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]], dtype=np.uint8)

kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], dtype=np.uint8)

eroded_image = erode(image, kernel)
print(eroded_image)
