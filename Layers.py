# The code file for layers

# from _typeshed import Self
import numpy as np
from numpy.core.fromnumeric import shape, size
from numpy.lib import math
from Utils import pad_with

# Layers(30 Points + 15 Bonus Points)
class Base():
    def __init__(self):

        pass

    # Empty
    def forward(self):
        pass

    def backward(self):
        pass


# Dense Layer(5 pts)
# X->hidden layers
class Dense(Base):
    def __init__(self, in_features,out_features):
        self.input = in_features
        self.N = out_features
        # Xavier Weight initialization with uniform distribution
        limit = np.sqrt(2 / float(self.input + self.N))
        self.weights = np.random.uniform(low=-limit, high=limit, size=(self.input, self.N))
        self.biases = np.random.uniform(low=-limit, high=limit, size=(self.N, 1))
        # Testing code
        self.gradient_weight_shape = (input, self.N)
        self.gradient_bias_shape = (self.N, 1)
        self.gradient_return = (input, self.N)
        self.dW = 0
        self.db = 0

    # N-number of neurons
    def forward(self, input_val):
        self.input_val = input_val
        if self.input_val.shape==3:
            self.input_val = self.input_val.reshape(input_val.shape[1]*input_val.shape[2],1)
        # F = X*W + b
        self.layer_output = np.dot(self.weights.T, self.input_val) + self.biases
        return self.layer_output

    def backward(self, delta_value):
        # Compute the gradient

        self.delta_value = delta_value
        self.dW += np.dot(self.input_val, self.delta_value.T)
        self.db += self.delta_value
        self.dx = np.dot(self.weights, self.delta_value)

        return self.dx

    def update(self):
        assert self.weights.shape == self.dW.shape
        self.weights = self.weights - (0.01/32) * self.dW
        assert self.biases.shape == self.db.shape
        self.biases = self.biases - (0.01/32) * self.db
        self.dW = np.zeros(self.dW.shape)
        self.db = np.zeros(self.db.shape)

# Convolutional Layer(10 pts)
# Conv(X, 10, (3x3), 1, 1) - - > takes X as inputsize,
#                               produces output with 10channels,
# #                               uses a 3 x 3 kernel with padding and stride both equal to 1.
class Conv(Base):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        # super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.backprop = np.zeros(out_channels,)



        self.new_kernal = np.random.randn(
            self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        # print(self.kernal_val.shape)

        self.gradient = np.zeros((self.new_kernal.shape))



    def forward(self,X):
        self.X = X
        # print("Input shape",self.X.shape)

        if self.padding > 0:
            self.padded_image = np.pad(self.X,((0,0),(self.padding,self.padding),(self.padding,self.padding)))
        else:
            self.padded_image = self.X


        self.new_width = int(math.floor((self.padded_image.shape[1]-self.new_kernal.shape[2])/self.stride) + 1)  # (W1−F)/S+1
        self.new_height = int(math.floor((self.padded_image.shape[2]-self.new_kernal.shape[3])/self.stride) + 1)  # (W2−F)/S+1
        
        self.backprop = np.zeros(self.new_kernal.shape)

        
        result = np.zeros((self.new_kernal.shape[0], self.new_width, self.new_height))

        a = 0
        b = self.new_kernal.shape[2]
        iter1 = 0
        for i in range(0, self.new_width):
            # a = i
            c = 0
            d = self.new_kernal.shape[3]
            iter2 = 0
            # check the less than condition
            while d <= self.padded_image.shape[2]:

                conv = self.padded_image[:,a:b,c:d]*self.new_kernal
                for z in range(0,self.new_kernal.shape[0]):
                    result[z][iter1][iter2] = np.sum(conv[z])
                

                c += self.stride
                d += self.stride

                iter2 += 1

            a += self.stride
            b = a+self.new_kernal.shape[2]
            iter1 += 1

        return result

    def backward(self, delta_value):
        #Convolving with dialted delta_value with kernal 
        self.delta_value = delta_value

        # Convolving with dialted delta_value with kernal   

        if self.stride > 1:
            add_val = self.stride - 1
            iter = 0
            iter += add_val
            self.dilated_delta = delta_value

            for i in range(0, self.dilated_delta.shape[1]-1):
                self.dilated_delta = np.insert(self.dilated_delta, iter, np.zeros(
                    (self.stride-1, self.stride-1)), -1)

                iter += add_val + 1

            iter = add_val
            for i in range(0, self.dilated_delta.shape[1]-1):
                self.dilated_delta = np.insert(self.dilated_delta, iter, np.zeros(
                    (self.stride-1, self.stride-1)), -2)

                iter += add_val + 1
    

        else:
            self.dilated_delta = self.delta_value
        
        
        self.padded_delta_value = np.pad(self.dilated_delta, 
        ((0, 0), (self.new_kernal.shape[3]-1, self.new_kernal.shape[3]-1), 
        (self.new_kernal.shape[3]-1, self.new_kernal.shape[3]-1)))


        a = 0
        b = self.dilated_delta.shape[2]
        iter1 = 0
        new_image = np.reshape(self.X,(1,self.X.shape[0],self.X.shape[1],self.X.shape[2]))
        # print("reshaped image shape",new_image.shape)
        self.new_dilated_delta = np.reshape(self.dilated_delta,(self.dilated_delta.shape[0],1,self.dilated_delta.shape[1],self.dilated_delta.shape[2]))
        # print("reshaped new_dilated_delta", self.new_dilated_delta.shape)
        for i in range(0, self.gradient.shape[1]):
            # a = i
            c = 0
            d = self.new_dilated_delta.shape[2]
            iter2 = 0
            # check the less than condition
            while b<self.X.shape[2] and d < self.X.shape[2]:
                conv = new_image[:, :, a:b, c:d]*self.new_dilated_delta
                self.gradient[:,:,iter1,iter2] = np.sum(conv, axis=(-2, -1))

                c += self.stride
                d += self.stride
                iter2 += 1
                # print("\n")
            # print("\t")
            a += self.stride
            b = a+self.dilated_delta.shape[2]
            iter1 += 1
        # print("gradients", self.gradient.shape)
        self.gradient += self.gradient

        
        self.padded_delta_value = np.pad(self.dilated_delta,((0, 0), (self.new_kernal.shape[3]-1, self.new_kernal.shape[3]-1), (self.new_kernal.shape[3]-1, self.new_kernal.shape[3]-1)))


        # Need to check the kernal inside value
        self.flip_kernal = np.flipud(np.fliplr(self.new_kernal))

        self.return_delta = np.zeros(self.X.shape)

        self.reshaped_delta = np.reshape(self.padded_delta_value, (self.padded_delta_value.shape[0], 1, self.padded_delta_value.shape[1], self.padded_delta_value.shape[2]))
        # print(self.reshaped_delta.shape)

        a = 0
        b = self.new_kernal.shape[2]
        iter1 = 0
        for i in range(0, self.return_delta.shape[1]):
            c = 0
            d = self.new_kernal.shape[2]
            iter2 = 0
            # check the less than condition
            while b < self.return_delta.shape[2] and d < self.return_delta.shape[2]:
                val = self.reshaped_delta[:, :, a:b, c:d]*self.new_kernal
  
                val = np.sum(val, axis=0)

                for z in range(0, self.new_kernal.shape[1]):
                    self.return_delta[z][iter1][iter2] = np.sum(val[z])

                c += self.stride
                d += self.stride
                iter2 += 1
                # print("\n")
            # print("\t")
            a += self.stride
            b = a+self.new_kernal.shape[2]
            iter1 += 1


        return self.return_delta



    def update(self):
        # assert self.weights.shape == self.dW.shape
        assert self.new_kernal.shape == self.gradient.shape
        self.new_kernal = self.new_kernal - (0.01/32) * self.gradient
        self.gradient = np.zeros(self.new_kernal.shape)
        # dynamic value for 32




# # Average Pooling(5 pts)
# # AvgPool(X, (2x2), 1, 1) - - > takes X as inputsize, uses a 2 x 2 kernel with padding and stride both equal to 1
class AvgPool(Base):
    def __init__(self, in_channels, kernel_size, stride, padding):
        # self.X = in_channels

        # TODO Change the shape of x(channel,_,_)

        self.no_of_channels = in_channels
        self.kernal_size = kernel_size
        self.stride = stride
        self.padding = padding


    def forward(self, X):

        self.X = X
        if self.padding > 0:
            self.padded_image = np.pad(
                self.X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)))

            # self.padded_image = np.zeros((self.no_of_in_channels, self.X .shape[1] + (2 * self.padding), self.X .shape[2]+(2 * self.padding)))
        else:
            self.padded_image = self.X


        self.activated_values = np.zeros(
            (self.padded_image.shape[0], self.padded_image.shape[1], self.padded_image.shape[2]))

        self.backward_val = np.ones(self.padded_image.shape)

        new_width = int(
            (self.padded_image.shape[1]-self.kernal_size[0])/self.stride + 1)  # (W1−F)/S+1
        new_height = int(
            (self.padded_image.shape[2]-self.kernal_size[1])/self.stride + 1)  # (W2−F)/S+1
        # print(new_width, new_height)
        self.pool_image = np.zeros(
            (self.no_of_channels, new_width, new_height))

        for channel in range(0, self.no_of_channels):
            a = 0
            b = self.kernal_size[0]
            for i in range(0, new_width):
                # a = i
                c = 0
                d = self.kernal_size[1]
                iter = 0
                while d <= self.padded_image.shape[2]:

                    max_val = np.average(self.padded_image[channel][a:b, c:d])
                    return_val = max_val / self.kernal_size[0]
                    self.backward_val[channel][a:b, c:d] *= return_val

                    c += self.stride
                    d += self.stride
                    self.pool_image[channel][i][iter] = max_val
                    # % += 1
                    iter += 1
                    # print("\n")
                # print("\t")
                a += self.stride
                b = a+self.kernal_size[1]

        return self.pool_image

    def backward(self, dz):
        # self.dz = dz
        self.dz = dz
        iter1 = 0
        result = np.zeros(self.backward_val.shape)
        # result = []
        for channel in range(0, self.backward_val.shape[0]):
            a = 0
            b = self.dz.shape[1]
            for i in range(0, self.backward_val.shape[1]):
                # a = i
                c = 0
                d = self.dz.shape[2]
                iter = 0
                while b < self.backward_val.shape[2] and d < self.backward_val.shape[2]:
                    temp = self.backward_val[channel][a:b, c:d]@dz[channel]

                    result[channel][a:b, c:d] = temp
                    c += self.stride
                    d += self.stride
                    # % += 1
                    iter += 1
                    # print("\n")
                # print("\t")
                a += self.stride
                b = a + self.kernal_size[1]
                # b = new_kernal.shape[0]

        return result
        # return self.backward_val

    def update(self):
        pass


# # Max Pooling(5 pts)
# # MaxPool(X,(2x2),1,1)  -- >takes X as inputsize, uses a 2 x 2 kernel with padding and stride both equal to 1
class MaxPool(Base):
    def __init__(self, in_channels, kernel_size, stride, padding):
        # super().__init__()
        # self.in_channels = in_channels

        # TODO Change the shape of x(channel,_,_)
        self.no_of_channels = in_channels
        self.kernal_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self,X):
        self.X = X

        if self.padding > 0:
            # self.padded_image = np.pad(self.X, self.padding, pad_with, padder="0")
            self.padded_image = np.pad(self.X,((0,0),(self.padding,self.padding),(self.padding,self.padding)))

        else:
            self.padded_image = self.X
        
        # print(self.padded_image.shape[1])
        # print(self.padded_image.shape[2])
        self.backward_val = np.zeros(self.padded_image.shape)


        self.activated_values = np.zeros((self.padded_image.shape[0], self.padded_image.shape[1], self.padded_image.shape[2]))

        new_width = int(math.floor((self.padded_image.shape[1]-self.kernal_size[0])/self.stride + 1))  # (W1−F)/S+1
        new_height = int(math.floor((self.padded_image.shape[2]-self.kernal_size[1])/self.stride + 1))  # (W2−F)/S+1
        # print(new_width, new_height)
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
                    max_val = np.max(self.padded_image[channel][a:b, c:d])
                    for_backward_temp = np.argwhere(
                        self.padded_image[channel][a:b, c:d] == max_val)
                    # print(max_val)
                    # print(for_backward_temp[0][0]+a)
                    # print(for_backward_temp[0][1]+c)
                    dimension_0 = for_backward_temp[0][0]+a
                    dimension_1 = for_backward_temp[0][1]+c
                    self.backward_val[channel][dimension_0][dimension_1] = 1
                    # print("\n")
                    c += self.stride
                    d += self.stride
                    pool_image[channel][i][iter] = max_val
                    # % += 1
                    iter += 1
                    # print("\n")
                # print("\t")
                a += self.stride
                b = a + self.kernal_size[1]

        return pool_image

    def backward(self, dz):
        # a = 0
        # b = self.backward_val.shape[2]
        self.dz = dz
        iter1 = 0
        result = np.zeros(self.backward_val.shape)
        # result = []
        for channel in range(0, self.backward_val.shape[0]):
            a = 0
            b = self.dz.shape[1]
            for i in range(0, self.backward_val.shape[1]):
                # a = i
                c = 0
                d = self.dz.shape[2]
                iter = 0
                while d <= self.backward_val.shape[2]:
                    # print(dz[channel].shape)
                    # print(self.backward_val[channel][a:b, c:d].shape)
                    temp = self.backward_val[channel][a:b, c:d]@dz[channel]

                    result[channel][a:b, c:d] = temp
                    c += self.stride
                    d += self.stride
                    # % += 1
                    iter += 1
                    # print("\n")
                # print("\t")
                a += self.stride
                b = a + self.kernal_size[1]

        return result

        # pass

        # return self.backward_val


    def update(self):
        # return np.multiply(self.backward_val, dz)
        pass


# Flatten Layer(5 pts)
# Flatten() -> takes the input and flattensthe last dimension
class Flatten(Base):
    def __init__(self):
        # super().__init__()
        pass

    def forward(self, X):
        self.X = X
        self.shape = self.X.shape
        self.flatten_val = self.X.flatten()
        self.flatten_val_reshape = self.flatten_val.reshape(self.X.shape[0]*self.X.shape[1]*self.shape[2], 1)
        return self.flatten_val_reshape

    def backward(self, delta):
        self.delta = delta
        self.original_shape = self.delta.reshape(self.X.shape[0], self.X.shape[1],self.X.shape[2])
        return self.original_shape
        
    def update(self):
        pass

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

    def forward(self, X):
        self.X = X
        # self.activated_neuorons = np.zeros(self.X.shape)
        
        return_val =  np.maximum(X, 0)
        self.activated_neuorons = np.where(return_val>0,1,0)
        return return_val

    def backward(self, val):

        return np.multiply(self.activated_neuorons, val)

    def update(self):
        pass


class Softmax(Base):
    def __init__(self):
        # super().__init__()
        pass

    def forward(self, X):
        self.X = X

        self.X = self.X - np.max(self.X)
        numerator = np.exp(self.X) + 1e-8
        # self.activated_neuorons = np.zeros(self.X.shape)
        denominator = np.sum(numerator)  # , axis=-1, keepdims=True)
        softmax_result = numerator / denominator
        self.val = softmax_result

        return softmax_result

    def backward(self, dz):
        #self.val = val
        self.activated_neuorons = np.zeros((dz.shape[0], dz.shape[0]))
        for i in range(len(self.val)):
            for j in range(len(self.val)):
                if i == j:
                    self.activated_neuorons[i, j] = self.val[i] * (1 - self.val[i])
                else:
                    self.activated_neuorons[i, j] = -self.val[i] * self.val[j]
        #To do
        #print(self.activated_neuorons)
        print(self.activated_neuorons.shape)
        return self.activated_neuorons @ dz

    def update(self):
        pass



# # Loss Functions and Metrics(18 points)
# # P -> prediction
# # Y -> ground truth
# def MSE(P,Y):
#     return np.mean((P-Y)**2)

class Softmax_CrossEntropy(Base):
    def __init__(self):
        # super().__init__()
        pass

    def forward(self, X, Y):
        self.X = X 
        self.Y = Y
        # softmax_result = (np.exp(self.X) / np.sum(np.exp(self.X), axis=0))
        shift = np.max(self.X, axis=0, keepdims=True)
        z = self.X - shift
        numerator = np.exp(z) + 1e-9
        denominator = np.sum(numerator)  # , axis=-1, keepdims=True)
        softmax_result = numerator / denominator
       # softmax_result = (np.exp(self.X) + 1e-9 )/ np.sum(np.exp(self.X), axis=0)
        # return softmax

        # CrossEntropy
        cross_entropy = -1 * (Y.T @ np.log(softmax_result))
        return softmax_result, cross_entropy

    def backward(self, P, Y):

        return P-Y


class CrossEntropy(Base):
    def __init__(self):
        # super().__init__()
        pass

    def forward(self, P,Y):
        self.P = P
        self.Y = Y

        cross_entropy = -1 * (Y.T @ np.log(P))
        return cross_entropy


    def backward(self,P,Y):
        return -Y/P


class Sigmoid(Base):
    def __init__(self):
        # super().__init__()
        pass

    def forward(self, X):
        self.X = X
        self.Z = 1.0 / (1.0 + np.exp(-self.X))
        return self.Z

    def backward(self, val):
        back = self.Z * (1 - self.Z) * val
        return back

    def update(self):
        pass


class MSE(Base):
    def __init__(self):
        pass
    def forward(self, P,Y):
        return (Y-P)**2

    def backward(self, P,Y):
        return -2*(Y-P)
