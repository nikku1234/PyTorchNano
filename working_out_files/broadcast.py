import numpy as np
img = np.random.random_integers(0, 255, (5, 5, 5))
kernal = np.random.random_integers(0, 255,(10,5,5, 5))
# in_channels = 5
# out_channels = 10
# new_kernal = np.tile(kernal, out_channels)
print((img*kernal).shape)