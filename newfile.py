import numpy as np

image = np.random.randint(2,3,(1,32,6,6))
dz = np.random.randint(4,7,(16,1,6,6))
# np.reshape(dz)
result = image * dz
# res = np.sum(image * dz,axis =0)
print(result.shape)
# print(image * dz)
# print((image*dz).shape)


