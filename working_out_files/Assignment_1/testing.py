import numpy as np
x = [1,0,-2]
for i in range(len(x)):
    if x[i]>0:
        x[i]=1
    else:
        x[i]= 0
print(x)