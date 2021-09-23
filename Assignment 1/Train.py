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

test_images = idx2numpy.convert_from_file(r"/Users/nikhil/Documents/GitHub/CSE-673-ComputationalVision/Assignment 1/Dataset/t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file(r"/Users/nikhil/Documents/GitHub/CSE-673-ComputationalVision/Assignment 1/Dataset/t10k-labels.idx1-ubyte")
train_images = idx2numpy.convert_from_file(r"/Users/nikhil/Documents/GitHub/CSE-673-ComputationalVision/Assignment 1/Dataset/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file(r"/Users/nikhil/Documents/GitHub/CSE-673-ComputationalVision/Assignment 1/Dataset/train-labels.idx1-ubyte")