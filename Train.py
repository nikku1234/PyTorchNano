from Layers import *
from Utils import *
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
    r"./Dataset/t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file(
    r"./Dataset/t10k-labels-idx1-ubyte")
train_images = idx2numpy.convert_from_file(
    r"./Dataset/train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file(
    r"./Dataset/train-labels-idx1-ubyte")

batch_size = 32 #Do not change, Using this to update the gradients
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


#Testing
test_labels_one_hot = []
for i in range(0,len(test_labels)):
    # print(train_labels[i])
    temp = np.zeros(10)
    temp[test_labels[i]] = 1
    test_labels_one_hot.append(temp)
    # print(temp)

soft_cross = Softmax_CrossEntropy()
softmax = Softmax()
cp = CrossEntropy()
accuracy = 0
# sigmoid =Sigmoid()
mse = MSE()
hinge = Hinge()

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

train_images = train_images/255
test_images = test_images/255

for i in range(epoch*2):
    predicted_index = []
    actual_index = []
    soft_cross_values = []
    if i % 2 == 0:
        for j in range(0,len(train_images)):
            # print(j)
            # if i % batch_size != 0:
            # print("forward")
            forward_output = model.forward(
                np.expand_dims(train_images[j], axis=0))

            cross_entropy_forward = cp.forward(forward_output,train_labels_one_hot[j].reshape(10, 1))
            #print(forward_output)

            # softmax_output = softmax.forward(forward_output)
            # print(softmax_output)
            #soft_cross_output,crossentropy_out = soft_cross.forward(forward_output, train_labels_one_hot[j].reshape((10, 1)))
            #print("crossentropy_out", crossentropy_out)
            #soft_cross_output_loss = soft_cross.backward(soft_cross_output, train_labels_one_hot[j].reshape((10,1)))
            #print(soft_cross_output_loss)

            #sigmoid_output = sigmoid.forward(forward_output)
            #print(sigmoid_output)
            #mse_forward_output = mse.forward(sigmoid_output, train_labels_one_hot[j].reshape((10, 1))) #loss
            #print(mse_forward_output)
            #mse_backward = mse.backward(soft_cross_output, train_labels_one_hot[j].reshape((10, 1)))
            #print(mse_backward)



            #Updates softmax and crossentropy
            #softmax_output = softmax.forward(forward_output)
            #print(forward_output)
            crossentropy_back_loss = cp.backward(forward_output, train_labels_one_hot[j].reshape(10, 1))
            predicted_index.append(list(forward_output).index(max(forward_output)))
            actual_index.append(list(train_labels_one_hot[j].reshape((10, 1))).index(1))
            model.backward(crossentropy_back_loss)







            #Appending predicted and actual indices for accuracy
            #predicted_index.append(list(soft_cross_output).index(max(soft_cross_output)))
            #actual_index.append(list(train_labels_one_hot[j].reshape((10,1))).index(1))

            #predicted_index.append(list(sigmoid_output).index(max(sigmoid_output)))
            #actual_index.append(list(train_labels_one_hot[j].reshape((10, 1))).index(1))
            # print("\nBackward")
            #model.backward(soft_cross_output_loss)

            #model.backward(mse_backward)
            # print("backward done")
            if (j% batch_size == 0 and j!=0):
                model.update()
                # print("updated")
            accuracy = Accuracy(predicted_index, actual_index)
            print(j)
            print("Train Accuracy: " + str(accuracy) + "%")
    else:
        for j in range(0, len(test_images)):
            # print(j)
            # if i % batch_size != 0:
            # print("forward")
            forward_output = model.forward(
                np.expand_dims(test_images[j], axis=0))
            # print(forward_output)

            softmax_output = softmax.forward(forward_output)
            # print(softmax_output)
            # soft_cross_output,crossentropy_out = soft_cross.forward(forward_output, test_labels_one_hot[j].reshape((10, 1)))
            # print("crossentropy_out", crossentropy_out)
            # soft_cross_output_loss = soft_cross.backward(soft_cross_output, test_labels_one_hot[j].reshape((10,1)))
            # print(soft_cross_output_loss)

            crossentropy_back_loss = cp.backward(softmax_output, train_labels_one_hot[j].reshape(10, 1))

            # Appending predicted and actual indices for accuracy
            # predicted_index.append(list(soft_cross_output).index(max(soft_cross_output)))
            # actual_index.append(list(test_labels_one_hot[j].reshape((10,1))).index(1))
            # soft_cross_values.append(soft_cross_output) #list of lists of predictions

            predicted_index.append(list(softmax_output).index(max(softmax_output)))
            actual_index.append(list(test_labels_one_hot[j].reshape((10, 1))).index(1))
            soft_cross_values.append(softmax_output)  # list of lists of predictions

            # print("updated")
            accuracy = Accuracy(predicted_index, actual_index)
            print(j)
            print(accuracy)
        confusion_matrix = ConfusionMatrix(predicted_index, actual_index)
        print("Confusion Matrix:")
        print(confusion_matrix)
        plot, AUC_Score = ROC(soft_cross_values, actual_index)
        plt.savefig(str(i) + 'ROC.png')
        print("AUC score:" + str(AUC_Score))
        print("Test Accuracy: " + str(accuracy) + "%")
