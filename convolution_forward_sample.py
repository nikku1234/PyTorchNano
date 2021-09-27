import numpy as np
from numpy.core.fromnumeric import shape, size

# image = np.random.randint(5, size = (10, 10))
# # print("here")
# kernal = np.random.randint(3, size = (3, 3))


# # image_sample = np.array([[-1, 0, 1],
# #                  [-1, 0, 1],
# #                  [-1, 0, 1]])


# # kernal_sample = np.array([[-1, 0, 1],
# #                  [-1, 0, 1],
# #                  [-1, 0, 1]])

# # print(np.sum(image_sample*kernal_sample))

# # print("shape:",image.shape)
# # print("image:\n",image[0:3,])
# # print("image[0:3]\n",image[0:3])
# # print("image[0:3][0:3]\n",image[0:3,0:3])
# # print(image[0:3,0:3].shape)
# # print(image[0:3,0:3])
# # print(image[0:3]*kernal[0:3])

# print("image\n",image)
# a = 0
# b = 3
# for i in range(0,10-3+1):
#     a = i
#     b = i+3
#     c = 0
#     d = 3
#     while c < (10-3+1):
#         print("a,b",a,b)
#         print(image[a:b,c:d],sep="/t")
#         c += 1
#         d += 1
#         # print("\t")
#     a += 1
#     b += 1



# print(image[0:3, 0:3]*kernal)


# new_image = np.array([[1,2,1,2,1,6],
# [2,1,3,4,5,6],
# [1,2,3,4,5,6],
# [2,1,2,1,2,3],
# [4,5,6,1,2,3],
# [1,2,1,2,1,2]]).reshape(6,6)

# print(new_image)

# new_kernal = np.array([[1,2,1],
# [1,2,1],
# [1,2,1]]).reshape(3,3)
# print(new_kernal)

# print("result")
# a = 0
# b = 3
# result = np.zeros((4,4))
# for i in range(0,6-3+1):
#     a = i
#     b = i+3
#     c = 0
#     d = 3
#     iter = 0
#     while c < (6-3+1):
#         print("a,b",a,b)
#         conv = np.sum(new_image[a:b, c:d]*new_kernal)
#         print(conv, sep="/t")
#         c += 1
#         d += 1
#         result[a][iter] = conv
#         iter += 1
#     print("\t")
#     a += 1
#     b += 1

# print("result",result)


# new_image = np.array([[1, 2, 1, 2, 1, 6],
#                       [2, 1, 3, 4, 5, 6],
#                       [1, 2, 3, 4, 5, 6],
#                       [2, 1, 2, 1, 2, 3],
#                       [4, 5, 6, 1, 2, 3],
#                       [1, 2, 1, 2, 1, 2]]).reshape(6, 6)

# print(new_image)

# new_kernal = np.array([[1, 2, 1],
#                        [1, 2, 1],
#                        [1, 2, 1]]).reshape(3, 3)
# print(new_kernal)
# stride = 1
# print("result")
# a = 0
# b = 3
# new_width = int((new_image.shape[0]-new_kernal.shape[0])/stride + 1)   #(W1−F)/S+1
# new_height = int((new_image.shape[1]-new_kernal.shape[1])/stride + 1)   #(W2−F)/S+1
# print(new_width,new_height)
# result = np.zeros((new_width, new_height))
# for i in range(0, new_width):
#     a = i
#     b = i+3
#     c = 0
#     d = 3
#     iter = 0
#     while c < (new_height):
#         print("a,b",a,b)
#         conv = np.sum(new_image[a:b, c:d]*new_kernal)
#         print(conv, sep="/t")
#         c += stride
#         d += stride
#         result[a][iter] = conv
#         iter += 1
#     print("\t")
#     a += 1
#     b += 1

# print("result",result)


new_image = np.array([[1, 0, 0, 2, 2],
                      [2, 2, 1, 0, 0],
                      [2, 0, 2, 1, 1],
                      [2, 1, 1, 2, 0],
                      [1, 2, 2, 0, 1]]).reshape(5, 5)

print(new_image)

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


padded_image = np.pad(new_image, 1, pad_with,padder="0")

print("padded image\n", padded_image)
print("new kernal\n", new_kernal)
stride = 2
print("result")
a = 0
b = new_kernal.shape[0]
new_width = int(
    (padded_image.shape[0]-new_kernal.shape[0])/stride + 1)  # (W1−F)/S+1
new_height = int(
    (padded_image.shape[1]-new_kernal.shape[1])/stride + 1)  # (W2−F)/S+1
print(new_width, new_height)
result = np.zeros((new_width, new_height))
for i in range(0, new_width):
    # a = i
    c = 0
    d = new_kernal.shape[1]
    iter = 0
    while d <= padded_image.shape[0]:
        print("a,b", a, b)
        print("c,d",c,d)
        print("multipying\n")
        print("padded image", padded_image[a:b, c:d])
        print("*")
        print("kernal", new_kernal)
        conv = np.sum(padded_image[a:b, c:d]*new_kernal)
        # print(conv, sep="/t")
        c += stride
        d += stride
        result[i][iter] = conv
        iter += 1
        print("\n")
    print("\t")
    a += stride
    b = a+new_kernal.shape[1]
    # b = new_kernal.shape[0]
    
    # b += stride

print("result", result)


