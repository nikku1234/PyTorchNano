# The code file for layers

import numpy as np

# Layers(30 Points + 15 Bonus Points)

# Dense Layer(5 pts)
# X->hidden layers
class Base():
    def __init__(self, input, N):
        self.N = N
        self.input = input
        limit = np.sqrt(2 / float(input + N))
        self.weights = np.random.uniform(low=-limit, high=limit, size=(input, N))
        self.biases = np.random.uniform(low=-limit, high=limit, size=(N, 1))


    def forward(input, N, weights, biases):
        return None

    def backward(dz):
        return None

class Dense(Base):
    def __init__(self, input, N):
        self.N = N
        self.input = input
        # Xavier Weight initialization with uniform distribution
        limit = np.sqrt(2 / float(input + N))
        self.weights = np.random.uniform(low = -limit, high = limit, size=(input, N))
        self.biases = np.random.uniform(low = -limit, high = limit, size=(N,))

    #N-number of neurons
    def forward(input, N, weights, biases):
        #f = XW + b
        layeroutput = np.dot(input, weights) + biases
        return layeroutput

    def backward(dz):
        gradient = None
        return gradient



# Convolutional Layer(10 pts)
# Conv(X, 10, (3x3), 1, 1) - - > takes X as inputsize, 
#                               produces output with 10channels, 
#                               uses a 3 x 3 kernel with padding and stride both equal to 1.
def Conv(X,no_of_channels,kernal_size,padding,stride):
    return None


# Average Pooling(5 pts)
# AvgPool(X, (2x2), 1, 1) - - > takes X as inputsize, uses a 2 x 2 kernel withpadding and stride both equal to 1
def AvgPool(X,kernal_size,padding,stride):
    return None


# Max Pooling(5 pts)
# MaxPool(X,(2x2),1,1)  -- >takes X as inputsize, uses a 2 x 2 kernel withpadding and stride both equal to 1
def MaxPool(X,kernal_size,padding,stride):
    return None


# Flatten Layer(5 pts)
# Flatten() -> takes the input and flattensthe last dimension
def Flatten():
    return None


# Dropout(5 bonus pts)
# Dropout(0.2) -> drops 20% of the inpu
def Dropout(percentage_of_input):
    return None


# BatchNorm(10 bonus pts):
# BatchNorm(X) ->takes X as input size
def BatchNorm(X):
    return None







# Activation functions(7 points)

def ReLU():
    return None


def Softmax():
    return None


def Sigmoid():
    return None






# Loss Functions and Metrics(18 points)
# P -> prediction
# Y -> ground truth
def MSE(P,Y):
    return np.mean((P-Y)**2)


def CrossEntropy(P,Y):
    return -log(1-P) #not sure


def Hinge(P,Y):
    return np.max(0, Y - (1-2*Y)*P)


# Output: A single value in %
def Accuracy(P,Y):
    return None


# Output: A 10 X 10 matrix
def ConfusionMatrix(P,Y):
    return None

# Output: A Plot using matplotlib of the ROC curve and Report the AUC score
def ROC(P,Y):
    return None
