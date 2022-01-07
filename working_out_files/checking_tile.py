import numpy as np
from numpy.lib.shape_base import tile

kernal_sample = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
kernal_sample = kernal_sample.reshape(3,3)
print(kernal_sample)
print(kernal_sample.shape)

tile_image = np.tile(kernal_sample,10)

print(tile_image)
print(tile_image.shape)
print(tile_image.reshape(10, 3, 3).shape)

print(np.random.randint(
    (10, 3, 3)))

# print(tile_image

