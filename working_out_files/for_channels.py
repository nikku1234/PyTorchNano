#working on channels
import numpy as np
import math
# image = np.random(size=(3,5,5))
img = np.random.random_integers(0, 255, (5, 5, 5))
kernal = np.random.random_integers(0, 5, (5, 3, 3))

new_kernal = kernal

# new_kernal = np.array([[[1, 0, 1],
#                        [0, 1, 0],
#                        [-1, 1, 0]],
#                       [[1, 0, 1],
#                        [0, 1, 0],
#                        [-1, 1, 0]],
#                       [[1, 0, 1],
#                        [0, 1, 0],
#                        [-1, 1, 0]]]).reshape(3, 3, 3)

# print(new_kernal[1])
no_of_channels = 5

print(img[0])
print(img[0][0])
print(img[0][0][0])

print("kernal shape:",kernal.shape)
# print(kernal[0])


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


padded_image = np.zeros((no_of_channels, 7, 7))

print(padded_image.shape[1])
print(padded_image.shape[2])




for i in range(0, no_of_channels):
    padded_image[i] = np.pad(img[i], 1, pad_with, padder="0")

    print("padded image\n", padded_image)
    print("new kernal\n", new_kernal)
    stride = 2
    print("result")

    new_width = int((padded_image.shape[1]-new_kernal.shape[1])/stride + 1)  # (W1−F)/S+1
    new_height = int((padded_image.shape[2]-new_kernal.shape[2])/stride + 1)  # (W2−F)/S+1
    print(new_width, new_height)
    result = np.zeros((no_of_channels,new_width, new_height))
    print("shape:",result.shape)

    for channel in range(0,no_of_channels):
        a = 0
        b = new_kernal.shape[1]
        for i in range(0, new_width):
            # a = i
            c = 0
            d = new_kernal.shape[2]
            iter = 0
            while d <= padded_image.shape[2]: #check the less than condition
                print("a,b", a, b)
                print("c,d", c, d)
                print("multipying\n")
                print("padded image", padded_image[channel][a:b, c:d])
                print("*")
                print("kernal", new_kernal[channel])
                conv = np.sum(padded_image[channel][a:b, c:d]*new_kernal[channel])
                # print(conv, sep="/t")
                c += stride
                d += stride
                result[channel][i][iter] = conv
                # % += 1
                iter += 1
                print("\n")
            print("\t")
            a += stride
            b = a+new_kernal.shape[2]
            # b = new_kernal.shape[0]
        # print("result", result)

        # b += stride

    print("result", result)
# print("result", result[0])
# print("result", result[0][0])
# print()











print("\n")
#Back Propogation
# new_result = np.insert(result[0], 1, np.zeros((1, 1)), 0)
# new_result = np.insert(new_result, 3, np.zeros((1, 1)), 0)
# new_result = np.insert(new_result, 1, np.zeros((1, 1)), 1)
# new_result = np.insert(new_result, 3, np.zeros((1, 1)), 1)


# print("\n",new_result)


stride = 3

checking = np.array([[1, -1],[-1, 1]]).reshape(2,2)
print(checking,"\n")
# new_checking = np.insert(checking, int(checking.shape[0]/2), np.zeros((stride-1, stride-1)), 0)
# print(new_checking)
# new_checking = np.insert(new_checking, int(checking.shape[1]/2), np.zeros((1, 1)), 1)
# new_checking = np.insert(new_checking, int(
#     checking.shape[1]/2), np.zeros((1, 1)), 1)

# print(new_checking)
# pad_res = np.pad(new_checking, 2, pad_with, padder="0")
# print("\n")
# print(pad_res)

# flip_kernal = np.flipud(np.fliplr(kernal[0]))
# print(flip_kernal)


new_result = np.insert(checking, int(checking.shape[0]/2), np.zeros((stride-1, stride-1)), 0)
i = stride-1
while i:
    new_result = np.insert(new_result, int(
        checking.shape[1]/2), np.zeros((1, 1)), 1)
    i -= 1

pad_res = np.pad(new_result, stride-1, pad_with, padder="0")
print("final array after the dilation\n",pad_res)
flip_kernal = np.flipud(np.fliplr(kernal[0]))
print(flip_kernal)
