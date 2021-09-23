from numpy.core.fromnumeric import reshape
from Layers_copy import *
import numpy as np

input = np.random.rand(5,5)
# print(input)
y = np.array([1,0])
input_flatten = np.squeeze(input)
input_flatten = input.reshape(25,1)
new1 = Dense(25,10)
forward1 = new1.forward(input_flatten)
print("Forward1 shape",forward1.shape)
relu1 = np.maximum(0,forward1)
new2 = Dense(10,2)
forward2 = new2.forward(relu1)
print("Forward2 shape",forward2.shape)
# print(forward2)
# softmax = Softmax()
# softmax_output = softmax.forward(forward2)
# print("Softmax Shape",softmax_output.shape)
y = reshape(2,1)
print("y shape ",y.shape)
soft_cross = Softmax_CrossEntropy()
soft_cross_output = soft_cross.forward(forward2,y)
soft_cross_output_back = soft_cross.backward(soft_cross_output,y)
print("nikhil", soft_cross_output.shape)

# gradient_val = np.random.rand(2,1)
gradeint2 = new2.backward(soft_cross_output_back)
gradeint1 = new1.backward(gradeint2)
print(gradeint1.shape)
# backward = new.backward(val)



# a = np.array([1,2,3,4,5]).reshape(1,5)
# b = np.array([1,2,3,4,5]).reshape(1,5)
# print(a.shape)
# print(np.dot(a.T,b))
