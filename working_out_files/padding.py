import numpy as np

new_image = np.array([[1, 0, 0, 2, 2],
                      [2, 2, 1, 0, 0],
                      [2, 0, 2, 1, 1],
                      [2, 1, 1, 2, 0],
                      [1, 2, 2, 0, 1]]).reshape(5, 5)

print(new_image.shape)

new_kernal = np.array([[1, 0, 1],
                       [0, 1, 0],
                       [-1, 1, 0]]).reshape(3, 3)

# new_image = np.random.randint(3, size=(25, 25))

# new_kernal = np.random.randint(3, size=(7, 3))

padding = 1

#numpy.org


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


padded_image = np.pad(new_image, padding, pad_with, padder="0")
print(padded_image.shape)
