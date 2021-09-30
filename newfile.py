import numpy as np

image = np.random.randint(2,3,(1,32,6,6))
dz = np.random.randint(4,7,(16,1,6,6))
# np.reshape(dz)
result = image * dz
# res = np.sum(image * dz,axis =0)
print(result.shape)
# print(image * dz)
# print((image*dz).shape)

stride = 2
new = np.ones((5,3,3))
# print(new)
# for i in range(new.shape[1]):
#     for j in range(new.shape[2]):
#         new[:, i, j] = np.insert(new, j, np.zeros((stride-1, stride-1)), -1)
#         j += stride-1
#     new[:, i, j] = np.insert(new, j, np.zeros((stride-1, stride-1)), -2)
#     i += stride-1
add_val = stride -1 
iter = add_val
for i in range(0,new.shape[1]-1):
    new = np.insert(new, iter, np.zeros((stride-1, stride-1)), -1)
    print(new.shape)
    print(new)
    iter += add_val + 1

iter = add_val
for i in range(0, new.shape[1]-1):
    new = np.insert(new, iter, np.zeros((stride-1, stride-1)), -2)
    print(new.shape)
    print(new)
    iter += add_val + 1

print(new)


# print(np.insert(new[0][0], 1,np.zeros((stride-1, stride-1)), 0))
