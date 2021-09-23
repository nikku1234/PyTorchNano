from Layers_copy import Softmax_CrossEntropy
from Model import create_model
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

test_images = idx2numpy.convert_from_file(
    r"/Users/nikhil/Documents/GitHub/CSE-673-ComputationalVision/Assignment_1/Dataset/t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file(
    r"/Users/nikhil/Documents/GitHub/CSE-673-ComputationalVision/Assignment_1/Dataset/t10k-labels-idx1-ubyte")
train_images = idx2numpy.convert_from_file(
    r"/Users/nikhil/Documents/GitHub/CSE-673-ComputationalVision/Assignment_1/Dataset/train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file(
    r"/Users/nikhil/Documents/GitHub/CSE-673-ComputationalVision/Assignment_1/Dataset/train-labels-idx1-ubyte")

batch_size = 32
epoch = 3

print()

#One-Hot Encode
train_labels_one_hot = []
for i in range(0,len(train_labels)):
    # print(train_labels[i])
    temp = np.zeros(10)
    temp[train_labels[i]] = 1
    train_labels_one_hot.append(temp)
    # print(temp)

# flatten = train_images[0].flatten().reshape(784, 1)
# print(flatten.shape)
# print(flatten.reshape(784,1).shape)
soft_cross = Softmax_CrossEntropy()

for i in range(epoch):
    for i in range(len(train_images)):
        # if i % batch_size != 0:
        print("forward")
        forward_output = model.forward(train_images[i].flatten().reshape(784, 1))
        # print(forward_output)

        soft_cross_output = soft_cross.forward(forward_output, train_labels_one_hot[i].reshape((10,1)))
        print("loss", soft_cross_output)
        soft_cross_output_loss = soft_cross.backward(soft_cross_output, train_labels_one_hot[i].reshape((10,1)))
        print("\nBackward")
        model.backward(soft_cross_output_loss)
        print("backward done")
        if (i+1) % batch_size == 0:
            model.update()
            print("updated")



    


