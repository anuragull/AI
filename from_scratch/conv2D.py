"""
import numpy as np

def conv2d(image, kernel):
    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2
    
    # Add zero padding to the image
    padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')
    
    # Create an output array
    output_height = image_height
    output_width = image_width
    output = np.zeros((output_height, output_width))
    
    # Perform the convolution operation
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return output

"""

import numpy as np

def max_pooling(image, pool_size):
    # Get dimensions of the image
    image_height, image_width = image.shape
    
    # Calculate output dimensions
    output_height = image_height // pool_size
    output_width = image_width // pool_size
    
    # Create an output array
    output = np.zeros((output_height, output_width))
    
    # Perform the max pooling operation
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.max(image[i*pool_size:i*pool_size+pool_size, j*pool_size:j*pool_size+pool_size])
    
    return output


def conv2d(image, kernel, stride=1, padding=0):
    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Compute output dimensions
    output_height = (image_height - kernel_height + 2 * padding) // stride + 1
    output_width = (image_width - kernel_width + 2 * padding) // stride + 1
    
    # Add zero padding to the image
    padded_image = np.pad(image, padding, mode='constant')
    
    # Create an output array
    output = np.zeros((output_height, output_width))
    
    # Perform the convolution operation
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(padded_image[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width] * kernel)
    
    return output


# Example input image and kernel
image = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]])

kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

# Perform convolution with stride=2 and padding=1
output = conv2d(image, kernel, stride=2, padding=1)





# Perform max pooling with pool size 2
pool_output = max_pooling(output, pool_size=2)

print("Max pooling output:")
print(pool_output)

# Print the output and its dimensions
print(output)
print(output.shape)
