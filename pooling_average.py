#Pooling
import numpy as np
#

kernal_size = (2, 2)
print(kernal_size[0])


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


#Max Pooling
# result = np.random.random_integers(0, 255, (1, 5, 5))
result = np.array([[1, 1, 2, 4],
                   [5, 6, 7, 8],
                   [3, 2, 1, 0],
                   [1, 2, 3, 4]]).reshape(1, 4, 4)

no_of_channels = result.shape[0]
padding = 0
stride = 2

if padding > 0:
    padded_image = np.zeros(
        (no_of_channels, result.shape[1] + (2 * padding), result.shape[2]+(2 * padding)))

    print(padded_image.shape[1])
    print(padded_image.shape[2])

    for i in range(0, no_of_channels):
        padded_image[i] = np.pad(result[i], padding, pad_with, padder="0")
else:
    padded_image = result

print("padded image\n", padded_image)

activated_values = np.zeros(
    (padded_image.shape[0], padded_image.shape[1], padded_image.shape[2]))
new_width = int(
    (padded_image.shape[1]-kernal_size[0])/stride + 1)  # (W1−F)/S+1
new_height = int(
    (padded_image.shape[2]-kernal_size[1])/stride + 1)  # (W2−F)/S+1
print(new_width, new_height)
pool_image = np.zeros((no_of_channels, new_width, new_height))


for channel in range(0, no_of_channels):
    a = 0
    b = kernal_size[0]
    for i in range(0, new_width):
        # a = i
        c = 0
        d = kernal_size[1]
        iter = 0
        while d <= padded_image.shape[2]:
            print("a,b", a, b)
            print("c,d", c, d)
            print("multipying\n")
            print("padded image", padded_image[channel][a:b, c:d])
            print("*")
            # print("kernal", new_kernal[channel])
            max_val = np.average(padded_image[channel][a:b, c:d])
            # conv = np.sum(padded_image[channel][a:b, c:d]*new_kernal[channel])
            # print(conv, sep="/t")
            c += stride
            d += stride
            pool_image[channel][i][iter] = max_val
            # % += 1
            iter += 1
            print("\n")
        print("\t")
        a += stride
        b = a+kernal_size[1]
        # b = new_kernal.shape[0]
    # print("result", result)

    # b += stride

print("result", pool_image)
# print("result", pool_image.shape)
# print("result", pool_image[0][0])
