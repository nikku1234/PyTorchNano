# The code file for layers

# from _typeshed import Self
import numpy as np
from numpy.core.fromnumeric import shape, size

# Layers(30 Points + 15 Bonus Points)
class Base():
    def __init__(self):
        # self.N = N
        # self.input = input
        # limit = np.sqrt(2 / float(input + N))  #
        # self.weights = np.random.uniform(low=-limit, high=limit, size=(input, N))
        # self.biases = np.random.uniform(low=-limit, high=limit, size=(N, 1))
        pass
    
    #Empty
    def forward(self):
        pass

    def backward(self):
        pass


# Dense Layer(5 pts)
# X->hidden layers
class Dense(Base):
    def __init__(self, input, N):
        self.input = input
        self.N = N
        # Xavier Weight initialization with uniform distribution
        limit = np.sqrt(2 / float(input + N))
        self.weights = np.random.uniform(low = -limit, high = limit, size=(input, N))
        self.biases = np.random.uniform(low = -limit, high = limit, size=(N,1))
        # Testing code
        self.gradient_weight_shape = (input,N)
        self.gradient_bias_shape = (N,1)
        self.gradient_return = (input,N)
        

    #N-number of neurons
    def forward(self,input_val):
        self.input_val = input_val
        # F = X*W + b
        self.layer_output = np.dot(self.weights.T, self.input_val) + self.biases
        return self.layer_output

    def backward(self, delta_value):
        #Compute the gradient
        self.new_ones = np.ones((self.N,1))
        self.delta_value = delta_value
        dW = np.dot(self.input_val,self.delta_value.T)
        db = np.dot(self.new_ones,self.delta_value.T)
        dx = np.dot(self.weights,self.delta_value)
        print("return of dx",dx.shape)
        return dx



# Convolutional Layer(10 pts)
# Conv(X, 10, (3x3), 1, 1) - - > takes X as inputsize, 
#                               produces output with 10channels, 
# #                               uses a 3 x 3 kernel with padding and stride both equal to 1.
# class Conv(Base):
#     def __init__(self, X, no_of_channels, kernal_size, padding, stride):
#         # super().__init__()
#         self.X = X
#         self.no_of_channels = no_of_channels
#         self.kernal_size = kernal_size
#         self.padding = padding
#         self.stride = stride

#     def forward(self):
#         pass

#     def backward(self):
#         pass
#         # return super().backward(x, y)


# # Average Pooling(5 pts)
# # AvgPool(X, (2x2), 1, 1) - - > takes X as inputsize, uses a 2 x 2 kernel withpadding and stride both equal to 1
# class AvgPool(Base):
#     def __init__(self, X, kernal_size, padding, stride):
#         self.X = X
#         self.kernal_size = kernal_size
#         self.padding = padding
#         self.stride = stride

#     def forward(self):
#         pass
#     def backward(self):
#         pass
#     # return None


# # Max Pooling(5 pts)
# # MaxPool(X,(2x2),1,1)  -- >takes X as inputsize, uses a 2 x 2 kernel withpadding and stride both equal to 1
# class MaxPool(Base):
#     def __init__(self,X,kernal_size,padding,stride):
#         # super().__init__()
#         self.X = X
#         self.kernal_size = kernal_size
#         self.padding = padding
#         self.stride = stride 
#     def forward(self):
#         pass
#     def backward(self):
#         pass


# Flatten Layer(5 pts)
# Flatten() -> takes the input and flattensthe last dimension
# class Flatten(Base):
#     def __init__(self,X):
#         # super().__init__()
#         return (self.X).flatten()



# # Dropout(5 bonus pts)
# # Dropout(0.2) -> drops 20% of the inpu
# def Dropout(percentage_of_input):
#     return None


# # BatchNorm(10 bonus pts):
# # BatchNorm(X) ->takes X as input size
# def BatchNorm(X):
#     return None



# Activation functions(7 points)

# class ReLU(Base):
#     def __init__(self):
#         pass
#     def forward(self,X):
#         self.X = X
#         return np.maximum(X,0)
#     def backward(self,val):
#         for i in range(len(self.X)):
#             if self.X[i] > 0:
#                 self.X[i] = 1
#             else:
#                 self.X[i] = 0
#         self.activated_neuorons = self.X
#         return np.dot(self.activated_neuorons,val)

class Softmax(Base):
    def __init__(self):
        # super().__init__()
        pass
    def forward(self,x):
        self.x = x
        return (np.exp(x) / np.sum(np.exp(x), axis=0))
    
    def backward(self,x):
        pass



class Sigmoid(Base):
    def __init__(self):
        # super().__init__()
        pass
    def forward(self,X):
        self.X = X
        self.Z = 1.0 / (1.0 + np.exp(-1.0*self.X))
        return self.Z
    def backward(self,val):
        return super().backward()



# # Loss Functions and Metrics(18 points)
# # P -> prediction
# # Y -> ground truth
# def MSE(P,Y):
#     return np.mean((P-Y)**2)


class CrossEntropy(Base):
    def __init__(self):
        pass
    def forward():
        pass
    def backward():
        pass

class Hinge(Base):
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass
# def Hinge(P,Y):
#     return np.max(0, Y - (1-2*Y)*P)


# # Output: A single value in %
# def Accuracy(P,Y):
#     return None


# # Output: A 10 X 10 matrix
# def ConfusionMatrix(P,Y):
#     return None

# # Output: A Plot using matplotlib of the ROC curve and Report the AUC score
# def ROC(P,Y):
#     return None