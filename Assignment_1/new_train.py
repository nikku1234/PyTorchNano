from Layers_copy import *
from Model_tests.Model_1 import create_model
import idx2numpy
import numpy as np
# from mnist import MNIST

# mnist = MNIST('../dataset/MNIST')
# x_train, y_train = mnist.load_training()  # 60000 samples
# x_test, y_test = mnist.load_testing()  # 10000 samples


#---------------------------------------------
# The following method would create the model
#---------------------------------------------
model = create_model()

test_images = idx2numpy.convert_from_file(r"/Users/nikhil/Documents/GitHub/CSE-673-ComputationalVision/Assignment_1/Dataset/t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file(
    r"/Users/nikhil/Documents/GitHub/CSE-673-ComputationalVision/Assignment_1/Dataset/t10k-labels-idx1-ubyte")
train_images = idx2numpy.convert_from_file(
    r"/Users/nikhil/Documents/GitHub/CSE-673-ComputationalVision/Assignment_1/Dataset/train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file(
    r"/Users/nikhil/Documents/GitHub/CSE-673-ComputationalVision/Assignment_1/Dataset/train-labels-idx1-ubyte")

batch_size = 32
epoch = 3

# print()

#One-Hot Encode
train_labels_one_hot = []
for i in range(0, len(train_labels)):
    # print(train_labels[i])
    temp = np.zeros(10)
    temp[train_labels[i]] = 1
    train_labels_one_hot.append(temp)
    # print(temp)

# flatten = train_images[0].flatten().reshape(784, 1)
# print(flatten.shape)
# print(flatten.reshape(784,1).shape)


soft_cross = Softmax_CrossEntropy()
softmax = Softmax()
accuracy = 0

# testing = np.array([[[1, 0, 1],
#                        [0, 1, 0],
#                        [-1, 1, 0]],
#                       [[1, 0, 1],
#                        [0, 1, 0],
#                        [-1, 1, 0]],
#                       [[1, 0, 1],
#                        [0, 1, 0],
#                        [-1, 1, 0]]]).reshape(3, 3, 3)
# forward_output = model.forward(np.expand_dims(testing, axis=0))
# print(forward_output.shape)

for i in range(epoch):
    predicted_index = []
    actual_index = []
    for j in range(0, len(train_images)):
        # print(j)
        # if i % batch_size != 0:
        # print("forward")
        # forward_output = model.forward(
        #     train_images[j].flatten().reshape(784, 1))
        #print(forward_output)

        forward_output = model.forward(np.expand_dims(train_images[j], axis=0))
        # print(forward_output.shape)

        # softmax_output = softmax.forward(forward_output)
        # print(softmax_output)
        soft_cross_output, crossentropy_out = soft_cross.forward(forward_output, train_labels_one_hot[j].reshape((10, 1)))
        #print("crossentropy_out", crossentropy_out)
        soft_cross_output_loss = soft_cross.backward(soft_cross_output, train_labels_one_hot[j].reshape((10, 1)))
        #print(soft_cross_output_loss)

        #Appending predicted and actual indices for accuracy
        predicted_index.append(list(soft_cross_output).index(max(soft_cross_output)))
        actual_index.append(list(train_labels_one_hot[j].reshape((10, 1))).index(1))
        # print("\nBackward")
        model.backward(soft_cross_output_loss)
        # print("backward done")
        if (j % batch_size == 0 and j != 0):
            model.update()
            # print("updated")
        accuracy = Accuracy(predicted_index, actual_index)
        print("Accuracy: " + str(accuracy) + "%")
