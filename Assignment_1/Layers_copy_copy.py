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
        # F = X*W + b
        self.layer_output = np.dot(self.weights.T, self.input_val) + self.biases
        return self.layer_output

    def backward(self, delta_value):
        # Compute the gradient
        # self.new_ones = np.ones((self.N,1))
        self.delta_value = delta_value
        self.dW += np.dot(self.input_val, self.delta_value.T)
        self.db += self.delta_value
        self.dx = np.dot(self.weights, self.delta_value)
        # print(self.dx)
        # print("return of dx",self.dx.shape)
        return self.dx

    def update(self):
        assert self.weights.shape == self.dW.shape
        self.weights = self.weights - (0.001/32) * self.dW
        assert self.biases.shape == self.db.shape
        self.biases = self.biases - (0.001/32) * self.db
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

        # print(self.out_channels)
        # print(self.kernel_size[0])

        self.kernal_val = np.random.randn(self.in_channels,self.kernel_size[0], self.kernel_size[1])
        # print(self.kernal_val.shape)

        #Change it to different values using random.
        self.new_kernal = np.tile(self.kernal_val, self.out_channels).reshape(
            self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])


    def forward(self,X):
        self.X = X
        print("Input shape",self.X.shape)

        if self.padding > 0:
            # self.padded_image = np.pad(self.X, self.padding, pad_with, padder="0")
            self.padded_image = np.pad(self.X,((0,0),(self.padding,self.padding),(self.padding,self.padding)))
        else:
            self.padded_image = self.X

        print("kernal shape:", self.kernel_size)
        print("padded image\n", self.padded_image.shape)
        print("new kernal\n", self.new_kernal.shape)

        self.new_width = int(
            (self.padded_image.shape[1]-self.new_kernal.shape[2])/self.stride + 1)  # (W1−F)/S+1

        self.new_height = int(
            (self.padded_image.shape[2]-self.new_kernal.shape[3])/self.stride + 1)  # (W2−F)/S+1
        print(self.new_width, self.new_height)
        self.backprop = np.zeros(self.new_kernal.shape)

        
        result = np.zeros((self.new_kernal.shape[0], self.new_width, self.new_height))
        print("Shape:", result.shape)

        # for channel in range(0, self.no_of_out_channels):
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
                print("a,b", a, b)
                print("c,d", c, d)
                # print("multipying\n")
                # print("padded image", self.padded_image[:,a:b,c:d].shape)
                # print("*")
                print("kernal", self.new_kernal.shape)
                conv = self.padded_image[:,a:b,c:d]*self.new_kernal
                for z in range(0,self.new_kernal.shape[0]):
                    result[z][iter1][iter2] = np.sum(conv[z])
                
                # print(conv.shape)
                # print(conv, sep="/t")
                c += self.stride
                d += self.stride
                # result[i][iter] = conv
                # % += 1
                iter2 += 1
                print("\n")
            print("\t")
            a += self.stride
            b = a+self.new_kernal.shape[2]
            iter1 += 1
            # b = new_kernal.shape[0]
        # print("result", result)

        # b += stride

        print("result", result.shape)
        # pass
        return result

    def backward(self, delta_value):
        #Convolving with dialted delta_value with kernal 
        
        self.delta_value = delta_value
        print(self.delta_value.shape)

        # Padding
        # if self.padding > 0:
            # self.padded_image = np.pad(self.X, self.padding, pad_with, padder="0")
       
        # else:
            # self.padded_delta_value = self.delta_value


        #Dilate
        if self.stride>1:
            temp_pad_res = []
            for i in range(self.delta_value.shape[0]):
                new_result = np.insert(self.delta_value[i], int(
                    self.delta_value.shape[1]/2), np.zeros((self.stride-1, self.stride-1)), 0)
                i = self.stride-1
                while i:
                    new_result = np.insert(new_result, int(new_result.shape[1]/2), np.zeros((1, 1)), 1)
                    i -= 1

                    temp_pad_res.append(new_result)
            self.pad_res = temp_pad_res
        else:
            self.pad_res = self.delta_value
        
        self.padded_delta_value = np.pad(self.pad_res,
        ((0, 0), (self.new_kernal.shape[3]-1, self.new_kernal.shape[3]-1), (self.new_kernal.shape[3]-1, self.new_kernal.shape[3]-1)))

        self.pad_res_delta = self.padded_delta_value
        # print(self.pad_res.shape)

        # print("final array after the dilation\n", self.pad_res)
        # Need to check the kernal inside value
        self.flip_kernal = np.flipud(np.fliplr(self.new_kernal))
        print(self.flip_kernal.shape)

        # new_width = int((self.pad_res.shape[1]-self.flip_kernal.shape[2])/self.stride + 1)  # (W1−F)/S+1
        # new_height = int((self.pad_res.shape[2]-self.flip_kernal.shape[3])/self.stride + 1)  # (W2−F)/S+1
        # print(new_width, new_height)
        back_result = np.zeros(self.X.shape)
        if delta_value.shape == self.X.shape:
            return np.zeros(self.X.shape)
        print("shape:", back_result.shape)


        for channel in range(0, self.out_channels):
            if channel > back_result.shape[0]:
                back_result = self.X
                break
            else:
                a = 0
                iter1 = 0
                b = self.flip_kernal.shape[2]
                for i in range(0, self.pad_res_delta.shape[1]):
                    # a = i
                    c = 0
                    d = self.flip_kernal.shape[3]
                    iter2 = 0
                    # check the less than condition
                    while b <= self.pad_res_delta.shape[2] and d <= self.pad_res_delta.shape[2]:
                        print("a,b", a, b)
                        print("c,d", c, d)
                        print("multipying\n")
                        print("padded image", self.pad_res_delta[channel][a:b, c:d])
                        print("*")
                        print("kernal", self.flip_kernal[channel])
                        conv = np.sum(
                            self.pad_res_delta[channel, a:b, c:d]*self.flip_kernal[channel])
                        # print(conv, sep="/t")
                        c += self.stride
                        d += self.stride
                        back_result[channel][iter1][iter2] = conv
                        # % += 1
                        iter2 += 1
                        print("\n")
                    print("\t")
                    a += self.stride
                    b = a+self.flip_kernal.shape[2]
                    iter1 += 1
                    # b = new_kernal.shape[0]
                # print("result", result)

                # b += stride

        print("result", back_result.shape) #backprop
        


        # Convolving with dialted delta_value with kernal
        if self.stride > 1:
            temp_pad_res = []
            for i in range(self.delta_value.shape[0]):
                new_result = np.insert(self.padded_delta_value[i], int(
                    self.padded_delta_value.shape[1]/2), np.zeros((self.stride-1, self.stride-1)), 0)
                i = self.stride-1
                while i:
                    new_result = np.insert(new_result, int(new_result.shape[1]/2), np.zeros((1, 1)), 1)
                    i -= 1

                    temp_pad_res.append(new_result)
            self.pad_res2 = np.array(temp_pad_res)
            print(self.pad_res2.shape)

            # self.pad_res2.res
        else:
            self.pad_res2 = self.delta_value

        # print("final array after the dilation\n", self.pad_res)
        # Need to check the kernal inside value
        # self.flip_kernal = np.flipud(np.fliplr(self.kernal[0]))
        # print(self.flip_kernal)

        # new_width = int((self.pad_res_2.shape[1]- self.pad_res_2.shape[1])/self.stride + 1)  # (W1−F)/S+1
        # new_height = int((self.pad_res_2.shape[2]- self.pad_res_2.shape[2])/self.stride + 1)  # (W2−F)/S+1
        # print(new_width, new_height)
        back_result_2 = np.zeros((self.new_kernal.shape))
        print("shape:", back_result_2.shape)

        for channel in range(0, self.out_channels):
            a = 0
            b = self.pad_res2.shape[1]
            iter1 = 0
            for i in range(0, self.X.shape[0]):
                # a = i
                c = 0
                d = self.pad_res2.shape[2]
                iter2 = 0
                # check the less than condition
                while b <= self.X.shape[2] and d <= self.pad_res2.shape[2]:
                    print("a,b", a, b)
                    print("c,d", c, d)
                    # print("multipying\n")
                    # print("padded image", self.pad_res_2[channel][a:b, c:d])
                    # print("*")
                    # print("kernal", self.flip_kernal[channel])
                    conv = np.sum(self.X[channel][a:b, c:d]
                                  * self.pad_res2[channel])
                    # print(conv, sep="/t")
                    c += self.stride
                    d += self.stride
                    back_result_2[channel][iter1][iter2] = conv
                    # % += 1
                    iter2 += 1
                    print("\n")
                print("\t")
                a += self.stride
                b = a+self.pad_res2.shape[2]
                iter1 += 1
                # b = new_kernal.shape[0]
            # print("result", result)

            # b += stride

        print("result", back_result_2) # save it and update
        self.backprop += back_result_2
        return back_result

    def update(self):
        # assert self.weights.shape == self.dW.shape
        self.weights = self.weights - (0.001/32) * self.dW  #dynamic value for 32

        
        # assert self.biases.shape == self.db.shape
        # self.biases = self.biases - (0.001/32) * self.db
        # self.dW = np.zeros(self.dW.shape)
        # self.db = np.zeros(self.db.shape)
    # def update(self):



        # pass

        # return super().backward(x, y)


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

        # if self.padding > 0:
        #     padded_image = np.zeros(
        #         (self.no_of_channels, self.X.shape[1] + (2 * self.padding), self.X.shape[2]+(2 * self.padding)))

        #     print(padded_image.shape[1])
        #     print(padded_image.shape[2])

        #     for i in range(0, self.no_of_channels):
        #         padded_image[i] = np.pad(
        #             self.X[i], padding, pad_with, padder="0")
        # else:
        #     padded_image = self.X

        # print("padded image\n", padded_image)

    def forward(self,X):

        self.X = X
        if self.padding > 0:
            self.padded_image = np.pad(self.X, self.padding, pad_with, padder="0")

                    # self.padded_image = np.zeros((self.no_of_in_channels, self.X .shape[1] + (2 * self.padding), self.X .shape[2]+(2 * self.padding)))
        else:
            self.padded_image = self.X
        
        print(self.padded_image.shape[1])
        print(self.padded_image.shape[2])
        self.activated_values = np.zeros((self.padded_image.shape[0], self.padded_image.shape[1], self.padded_image.shape[2]))

        self.backward_val = np.ones(self.padded_image.shape)

        new_width = int(
            (self.padded_image.shape[1]-self.kernal_size[0])/self.stride + 1)  # (W1−F)/S+1
        new_height = int(
            (self.padded_image.shape[2]-self.kernal_size[1])/self.stride + 1)  # (W2−F)/S+1
        print(new_width, new_height)
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
                    print("a,b", a, b)
                    print("c,d", c, d)
                    print("multipying\n")
                    print("padded image", self.padded_image[channel][a:b, c:d])
                    print("*")
                    # print("kernal", new_kernal[channel])
                    max_val = np.average(self.padded_image[channel][a:b, c:d])
                    return_val = max_val / self.kernal_size[0]
                    self.backward_val[channel][a:b, c:d] *= return_val
                    print(return_val)
                    # conv = np.sum(padded_image[channel][a:b, c:d]*new_kernal[channel])
                    # print(conv, sep="/t")
                    c += self.stride
                    d += self.stride
                    self.pool_image[channel][i][iter] = max_val
                    # % += 1
                    iter += 1
                    print("\n")
                print("\t")
                a += self.stride
                b = a+self.kernal_size[1]
                # b = new_kernal.shape[0]
            # print("result", result)

            # b += stride

        print("result of forward propogation avg pool", self.pool_image)
        return self.pool_image

    def backward(self,dz):
        self.dz = dz
        print("result backward avg pool", self.backward_val)
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
                    temp = self.backward_val[channel][a:b, c:d]@dz[channel]
                    # for_backward_temp = np.argwhere(
                    #     self.padded_image[channel][a:b, c:d] == max_val)
                    # print(max_val)
                    # print(for_backward_temp[0][0]+a)
                    # print(for_backward_temp[0][1]+c)
                    # dimension_0 = for_backward_temp[0][0]+a
                    # dimension_1 = for_backward_temp[0][1]+c
                    # self.backward_val[channel][dimension_0][dimension_1] = 1
                    print("\n")
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
            # print("result", result)

            # b += stride

        # print("result", pool_image)
        print("result", result.shape)
        # print("result", pool_image[0][0])
        return result
        # return self.backward_val


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
        
        print(self.padded_image.shape[1])
        print(self.padded_image.shape[2])
        self.backward_val = np.zeros(self.padded_image.shape)


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
                    max_val = np.max(self.padded_image[channel][a:b, c:d])
                    for_backward_temp = np.argwhere(
                        self.padded_image[channel][a:b, c:d] == max_val)
                    print(max_val)
                    print(for_backward_temp[0][0]+a)
                    print(for_backward_temp[0][1]+c)
                    dimension_0 = for_backward_temp[0][0]+a
                    dimension_1 = for_backward_temp[0][1]+c
                    self.backward_val[channel][dimension_0][dimension_1] = 1
                    print("\n")
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

        # print("result", pool_image)
        # print("result", pool_image.shape)
        # print("result", pool_image[0][0])
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
                    temp = self.backward_val[channel][a:b, c:d]@dz[channel]
                    # for_backward_temp = np.argwhere(
                    #     self.padded_image[channel][a:b, c:d] == max_val)
                    # print(max_val)
                    # print(for_backward_temp[0][0]+a)
                    # print(for_backward_temp[0][1]+c)
                    # dimension_0 = for_backward_temp[0][0]+a
                    # dimension_1 = for_backward_temp[0][1]+c
                    # self.backward_val[channel][dimension_0][dimension_1] = 1
                    print("\n")
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
            # print("result", result)

            # b += stride

        # print("result", pool_image)
        print("result", result.shape)
        # print("result", pool_image[0][0])
        return result

        # pass

        # return self.backward_val



        # return np.multiply(self.backward_val, dz)
        # pass


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
        # for i in range(len(self.X)):
        #     if self.X[i] > 0:
        #         self.activated_neuorons[i] = 1
        #     else:
        #         self.activated_neuorons[i] = 0
        # self.activated_neuorons = self.X
        # print("shape of relu backward", np.multiply(self.activated_neuorons,val).shape)
        return np.multiply(self.activated_neuorons, val)

    def update(self):
        pass


class Softmax(Base):
    def __init__(self):
        # super().__init__()
        pass

    def forward(self, x):
        self.x = x
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def backward(self, x):
        pass


class Sigmoid(Base):
    def __init__(self):
        # super().__init__()
        pass

    def forward(self, X):
        self.X = X
        self.Z = 1.0 / (1.0 + np.exp(-1.0*self.X))
        return self.Z

    def backward(self, val):
        return (self.Z * (1-self.Z))


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

    def backward(self, predicted, Y):

        # print("lolzmaooo")
        # print(predicted)
        # print("lulz")
        # print(Y)
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

    def forward(self, P, Y):
        return np.max(0, Y - (1-2*Y)*P)

    def backward(self):
        pass
# def Hinge(P,Y):
#     return np.max(0, Y - (1-2*Y)*P)


# # Output: A single value in %
def Accuracy(P, Y):
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
