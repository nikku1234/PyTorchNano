import numpy as np

dz = np.ones((16,8,8))
print(len(dz.shape))

stride = 3

# print(np.pad(dz,((0,0),(5,5),(5,5))).shape)

if stride > 1:
    for i in range(dz.shape[0]):
        new_result = np.insert(dz[i], int(dz.shape[1]/2), np.zeros((stride-1, stride-1)), 0)
        # i = stride-1
        # print(new_result)
        while i:
            new_result = np.insert(new_result, int(new_result.shape[1]/2), np.zeros((1, 1)), 1)
            i -= 1
            print(new_result)


            # self.pad_res[i] = np.pad(new_result, self.stride-1, pad_with, padder="0")
else:
    pad_res = dz
