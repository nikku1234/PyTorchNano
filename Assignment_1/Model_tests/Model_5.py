from GraphNet import Net
from Layers import *

def create_model():
	Model = Net()
	Model.add(AvgPool(in_channels=1, kernel_size=(2,2), stride=2, padding=0))
	Model.add(MaxPool(in_channels=1, kernel_size=(3,3), stride=3, padding=1))
	Model.add(ReLU())
	Model.add(Flatten())
	Model.add(Dense(in_features=25, out_features=32))
	Model.add(ReLU())
	Model.add(Dense(in_features=32, out_features=10))
	Model.add(Softmax())
	return Model
