import cv2
import numpy as np
from scipy.linalg import toeplitz


def matrix_to_vector(input):
    input_h, input_w = input.shape
    output_vector = np.zeros(input_h*input_w, dtype=input.dtype)

    input = np.flipud(input)
    for i,row in enumerate(input):
        st = i * input_w
        nd = st + input_w
        output_vector[st:nd] = row

    return output_vector

def vector_to_matrix(input, output_shape):
    output_h, output_w = output_shape
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_h):
        st = i * output_w
        nd = st + output_w
        output[i,:] = input[st:nd]
    output = np.flipud(output)   

    return output

img = cv2.imread('rsz2.png', cv2.IMREAD_GRAYSCALE)

kernel = np.array(([1,2,1],
                  [2,4,2],
                  [1,2,1]))

kernel = kernel/16
img_h, img_w = img.shape


kernel_x, kernel_y = kernel.shape

padding_x = (kernel_x - 1) // 2
padding_y = (kernel_y - 1) // 2


output_x = img_h + kernel_x - 1
output_y = img_w + kernel_y - 1

kernel_zero_padded = np.pad(kernel, ((output_x-kernel_x, 0),(0, output_y-kernel_y)), 'constant', constant_values=0)

toeplitz_list = []

for i in range(kernel_zero_padded.shape[0]-1,-1,-1):
    c = kernel_zero_padded[i,:]
    r = np.r_[c[0], np.zeros(img_w-1, dtype=int)]
    toeplitz_matrix = np.zeros((kernel_zero_padded.shape[0], img.shape[1]))
    for j in range(toeplitz_matrix.shape[0]-kernel_x+1):
        toeplitz_matrix[j:j+kernel_x,j] += c[0:kernel_y]
    toeplitz_list.append(toeplitz_matrix)

c = range(1, kernel_zero_padded.shape[0] + 1)
c = np.array(c)
doubly_indices = np.zeros((c.shape[0],img.shape[0]), dtype=int)
for i in range(img.shape[0]):
    doubly_indices[i:c.shape[0],i] += c[0:c.shape[0]-i]
    
h = toeplitz_matrix.shape[0] * doubly_indices.shape[0]
w = toeplitz_matrix.shape[1] * doubly_indices.shape[1]
doubly_blocked_shape = [h, w]
doubly_blocked = np.zeros(doubly_blocked_shape)
b_h, b_w = toeplitz_matrix.shape

for i in range(doubly_indices.shape[0]):
    for j in range(doubly_indices.shape[1]):
        start_i = i * b_h
        start_j = j * b_w
        end_i = start_i + b_h
        end_j = start_j + b_w
        doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

input_vector = matrix_to_vector(img)
output_vector = np.matmul(doubly_blocked, input_vector)

output_image = vector_to_matrix(output_vector,(output_x,output_y))

output_image = cv2.normalize(output_image,None,0,1,cv2.NORM_MINMAX)
cv2.imshow('output', output_image)

cv2.waitKey(0)

cv2.destroyAllWindows()