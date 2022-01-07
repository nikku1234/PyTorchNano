import numpy as np
import matplotlib.pyplot as plt
import copy
# For defining any helper functions and other metrics

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

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

        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)

        X_coordinates.append(FPR)
        Y_coordinates.append(TPR)
    #print(X_coordinates)
    #print(Y_coordinates)
    plt.plot(X_coordinates, Y_coordinates)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    #plt.show()
    newtpr = []
    newfpr = []
    for i in range(len(X_coordinates)-1):
        newtpr.append([Y_coordinates[i], Y_coordinates[i+1]])
        newfpr.append([X_coordinates[i], X_coordinates[i + 1]])
    auc = sum(np.trapz(newfpr, newtpr))+1
    return plt, auc



