#Stackoverflow

import numpy as np
from numpy.lib.stride_tricks import as_strided


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


# padded_image = np.pad(new_image, 1, pad_with, padder="0")


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    # A = np.pad(A, padding, mode='constant')

    padded_image = np.pad(A, padding, pad_with)

    # Window view of A
    output_shape = ((padded_image.shape[0] - kernel_size)//stride + 1,
                    (padded_image.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride*A.strides[0],
                              stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


# Example:

# >> > A = np.array([[1, 1, 2, 4],
#                   [5, 6, 7, 8],
#                   [3, 2, 1, 0],
#                   [1, 2, 3, 4]])

# >> > pool2d(A, kernel_size=2, stride=2, padding=0, pool_mode='max')

# array([[6, 8],
#        [3, 4]])
