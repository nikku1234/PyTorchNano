import numpy as np
result = np.array([[1, 1, 2, 4],
                   [5, 6, 7, 8],
                   [3, 2, 1, 0],
                   [1, 2, 3, 4]]).reshape(1, 4, 4)
max_val = np.average(result[0][0:2, 0:2])
new_result = np.zeros((1,4,4))
print(max_val)
print(np.where(result[0][0:2, 0:2] == max_val, new_result[0][0: 2, 0: 2], 1))
# print(np.where(result[0][0:2,2:4]==8))
print(result[0][0:2,2:4])
# print(result[1][1])

a = [3,1,4]
b = [1]
print(a+b)