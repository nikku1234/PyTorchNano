# The code file for layers

# from _typeshed import Self
import numpy as np
from numpy.core.fromnumeric import shape, size
from Utils import pad_with

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
        self.dW = 0
        self.db = 0
        

    #N-number of neurons
    def forward(self,input_val):
        self.input_val = input_val
        # F = X*W + b
        self.layer_output = np.dot(self.weights.T, self.input_val) + self.biases
        return self.layer_output

    def backward(self, delta_value):
        #Compute the gradient
        # self.new_ones = np.ones((self.N,1))
        self.delta_value = delta_value
        self.dW += np.dot(self.input_val,self.delta_value.T)
        self.db += self.delta_value
        self.dx = np.dot(self.weights,self.delta_value)
        #print(self.dx)
        # print("return of dx",self.dx.shape)
        return self.dx

    def update(self):
        assert self.weights.shape == self.dW.shape
        self.weights = self.weights - (0.001/32) * self.dW
        assert self.biases.shape == self.db.shape
        self.biases = self.biases - (0.001/32) * self.db
        self.dW = np.zeros(self.dW.shape)
        self.db = np.zeros(self.db.shape)
    # def update(self):
    #     self.dW /= batchsize



# Convolutional Layer(10 pts)
# Conv(X, 10, (3x3), 1, 1) - - > takes X as inputsize, 
#                               produces output with 10channels, 
# #                               uses a 3 x 3 kernel with padding and stride both equal to 1.
class Conv(Base):
    def __init__(self, X, no_of_channels, kernal_size, padding, stride):
        # super().__init__()
        self.X = X
        self.no_of_channels = no_of_channels
        self.kernal_size = kernal_size
        self.padding = padding
        self.stride = stride

    def forward(self):
        print("kernal shape:", self.kernal_size)
        
        
        pass

    def backward(self):
        pass
        # return super().backward(x, y)


# # Average Pooling(5 pts)
# # AvgPool(X, (2x2), 1, 1) - - > takes X as inputsize, uses a 2 x 2 kernel with padding and stride both equal to 1
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
# # MaxPool(X,(2x2),1,1)  -- >takes X as inputsize, uses a 2 x 2 kernel with padding and stride both equal to 1
class MaxPool(Base):
    def __init__(self,X,kernal_size,padding,stride):
        # super().__init__()
        self.X = X
        
        #Change the shape of x(channel,_,_)

        self.no_of_channels = X.shape(0)
        self.kernal_size = kernal_size
        self.padding = padding
        self.stride = stride 
        # self.backward_val = np.zeros(self.X.shape)

        if self.padding > 0:
            self.padded_image = np.zeros((self.no_of_channels, self.X .shape[1] + (2 * self.padding), self.X .shape[2]+(2 * self.padding)))

        else:
            self.padded_image = self.X

        print(self.padded_image.shape[1])
        print(self.padded_image.shape[2])
        self.backward_val = np.zeros(self.padded_image.shape)


    def forward(self):
        self.activated_values = np.zeros((self.padded_image.shape[0], self.padded_image.shape[1], self.padded_image.shape[2]))


        new_width = int((self.padded_image.shape[1]-self.kernal_size[0])/self.stride + 1)  # (W1−F)/S+1
        new_height = int((self.padded_image.shape[2]-self.kernal_size[1])/self.stride + 1)  # (W2−F)/S+1
        print(new_width, new_height)
        pool_image = np.zeros((self.no_of_channels, new_width, new_height))


        for channel in range(0, self.no_of_channels):
            a = 0
            b = self.kernal_size[0]
            for i in range(0, new_width):
                # a = i
                c = 0
                d = self.kernal_size[1]
                iter = 0
                while d <= self.padded_image.shape[2]:
                    # print("a,b", a, b)
                    # print("c,d", c, d)
                    # print("multipying\n")
                    # print("padded image", padded_image[channel][a:b, c:d])
                    # print("*")
                    # print("kernal", new_kernal[channel])
                    max_val = np.max(self.padded_image[channel][a:b, c:d])
                    for_backward_temp = np.argwhere(self.padded_image[channel][a:b, c:d] == max_val)
                    print(max_val)
                    print(for_backward_temp[0][0]+a)
                    print(for_backward_temp[0][1]+c)
                    dimension_0 = for_backward_temp[0][0]+a
                    dimension_1 = for_backward_temp[0][1]+c
                    self.backward_val[channel][dimension_0][dimension_1] = max_val
                    print("\n")
                    # conv = np.sum(padded_image[channel][a:b, c:d]*new_kernal[channel])
                    # print(conv, sep="/t")
                    c += self.stride
                    d += self.stride
                    pool_image[channel][i][iter] = max_val
                    # % += 1
                    iter += 1
                    # print("\n")
                # print("\t")
                a += self.stride
                b = a + self.kernal_size[1]
                # b = new_kernal.shape[0]
            # print("result", result)

            # b += stride

        print("result", pool_image)
        # print("result", pool_image.shape)
        # print("result", pool_image[0][0])
        return pool_image
    def backward(self):
        return self.backward_val
        pass


# Flatten Layer(5 pts)
# Flatten() -> takes the input and flattensthe last dimension
class Flatten(Base):
    def __init__(self):
        # super().__init__()
        pass

    def forward(self,X):
        self.X = X
        self.shape = self.X.shape
        self.flatten_val = self.X.flatten()
        self.flatten_val_reshape = self.flatten_val.reshape(self.shape[0]*self.shape[1],1)
        return self.flatten_val_reshape
    
    def backward(self,return_x):
        self.return_x = return_x
        self.original_shape = self.return_x.reshape(self.shape[0],self.shape[1])
        return self.original_shape



# # Dropout(5 bonus pts)
# # Dropout(0.2) -> drops 20% of the inpu
# def Dropout(percentage_of_input):
#     return None


# # BatchNorm(10 bonus pts):
# # BatchNorm(X) ->takes X as input size
# def BatchNorm(X):
#     return None



# Activation functions(7 points)

class ReLU(Base):
    def __init__(self):
        pass
    def forward(self,X):
        self.X = X
        self.activated_neuorons = np.zeros(self.X.shape)
        return np.maximum(X,0)
    def backward(self,val):
        for i in range(len(self.X)):
            if self.X[i] > 0:
                self.activated_neuorons[i] = 1
            else:
                self.activated_neuorons[i] = 0
        # self.activated_neuorons = self.X
        # print("shape of relu backward", np.multiply(self.activated_neuorons,val).shape)
        return np.multiply(self.activated_neuorons, val)
    def update(self):
        pass

class Softmax(Base):
     def __init__(self):
         # super().__init__()
         pass
     def forward(self,x):
         self.x = x
         return np.exp(x) / np.sum(np.exp(x), axis=0)
    
     def backward(self,x):
         pass



# class Sigmoid(Base):
#     def __init__(self):
#         # super().__init__()
#         pass
#     def forward(self,X):
#         self.X = X
#         self.Z = 1.0 / (1.0 + np.exp(-1.0*self.X))
#         return self.Z
#     def backward(self,val):
#         return super().backward()



# # Loss Functions and Metrics(18 points)
# # P -> prediction
# # Y -> ground truth
# def MSE(P,Y):
#     return np.mean((P-Y)**2)

class Softmax_CrossEntropy(Base):
    def __init__(self):
        # super().__init__()
        pass
    def forward(self,X,Y):
        self.X = X + 1e-9
        self.Y = Y
        # softmax_result = (np.exp(self.X) / np.sum(np.exp(self.X), axis=0))
        z = self.X - np.max(self.X, axis=-1, keepdims=True)
        numerator = np.exp(z)
        denominator = np.sum(numerator)#, axis=-1, keepdims=True)
        softmax_result = numerator / denominator
        softmax_result = np.exp(self.X) / np.sum(np.exp(self.X), axis=0)
        # return softmax

        # CrossEntropy
        cross_entropy = -1 * (Y.T @ np.log(softmax_result))
        return softmax_result,cross_entropy


    def backward(self,predicted,Y):

        #print("lolzmaooo")
        #print(predicted)
        #print("lulz")
        #print(Y)
        # self.predicted = predicted
        # self.Y = Y
        return predicted-Y


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
def Accuracy(P,Y):
    count = 0
    total = 0
    for i in range(len(P)):
        if P[i] == Y[i]:
            count = count + 1
        total = total + 1
        # total = total + 1
    return (count/total)*100


# # Output: A 10 X 10 matrix
# def ConfusionMatrix(P,Y):
#     return None

# # Output: A Plot using matplotlib of the ROC curve and Report the AUC score
# def ROC(P,Y):
#     return None
