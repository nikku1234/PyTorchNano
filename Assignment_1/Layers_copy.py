# The code file for layers

# from _typeshed import Self
import matplotlib.pyplot as plt
import copy
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
        
        pass

    def backward(self):
        pass
        # return super().backward(x, y)


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
        print
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
def ConfusionMatrix(P,Y):
    #X-axis true class
    #Y-axis Predicted class
    cm = np.zeros((10, 10))
    for i in range(len(P)):
        if P[i] == Y[i]:
            cm[P[i]][P[i]] = cm[P[i]][P[i]] + 1
        else:
            cm[P[i]][Y[i]] = cm[P[i]][Y[i]] + 1
    #print(cm)
    return cm

# # Output: A Plot using matplotlib of the ROC curve and Report the AUC score
def ROC(P,Y):
    thresholds = [0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1]
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    X_coordinates = []
    Y_coordinates = []
    for j in thresholds:
        pred = copy.deepcopy(P)
        orig = copy.deepcopy(Y)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        #convert each P[k] to zeroes and ones
        for k in range(len(pred)):
            for i in range(len(pred[k])):
                if pred[k][i] >= j:
                    pred[k][i] = 1
                else:
                    pred[k][i] = 0
        #print(P)
        for k in range(len(pred)):
            for i in range(len(pred[k])):
                numberorg = orig[k]
                numberpre = i
                if pred[k][i] == 1:
                    if numberorg == numberpre:
                        TP = TP + 1
                    else:
                        FP = FP +1
                elif pred[k][i] == 0:
                    if numberorg == numberpre:
                        FN = FN + 1
                    else:
                        TN =TN + 1
        #print(TP)
        #print(FP)
        #print(TN)
        #print(FN)
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        #print(TPR)
        #print(FPR)
        #plt.plot(FPR, TPR)
        #plt.show()
        X_coordinates.append(FPR)
        Y_coordinates.append(TPR)
        '''for val in range(0,10):
            for k in range(len(P)):
                for n in range(len(P[k])):
                    if P[k][n] > j:
                        number = list(P[k]).index(P[k][n])
                        if number == Y[k] == val:
                            TP = TP + 1
                        elif P[k][n] == 0:
                            TN =TN + 1
                        else:
                            if number == 0:
                                FN =FN +1
                            else:
                                FP = FP + 1
            TPR = TP/(TP+FN)
            FPR = FP/(FP+TN)
            plt.scatter(FPR,TPR)

        for i in range(len(P)):
            if max(P[i]) > j:
                number = list(P[i]).index(max(P[i]))
                if number == Y[i]:
                    matrix[number][number] = matrix[number][number] + 1
                else:
                    matrix[number][Y[i]] = matrix[number][Y[i]] + 1
        for k in range(0,10):
            TP = matrix[k][k]
            FP = sum(matrix[k]) - matrix[k][k]
            FN = sum([i[k] for i in matrix]) - matrix[k][k]
            x = np.matrix(matrix)
            x_sum = x.sum()
            TN = x_sum - (sum([i[k] for i in matrix]) + sum(matrix[k]) - matrix[k][k])
            TPR.append(TP/(TP+FN))
            FPR.append(FP/(FP+TN))
        #finding the macro average of all classes
        #TPRdict[j] = np.mean(TPR)
        #FPRdict[j] = np.mean(FPR)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    for key, value in TPRdict.items():
        plt.scatter(FPRdict[key],value)
        #print(value)
        #print(FPRdict[key])'''
    #print(X_coordinates)
    #print(Y_coordinates)
    #plt.plot(X_coordinates, Y_coordinates)
    #plt.show()
    newtpr = []
    newfpr = []
    for i in range(len(X_coordinates)-1):
        newtpr.append([Y_coordinates[i], Y_coordinates[i+1]])
        newfpr.append([X_coordinates[i], X_coordinates[i + 1]])
    auc = sum(np.trapz(newfpr, newtpr))+1
    return auc
