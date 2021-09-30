from GraphNet import Net
from Layers_copy import *

def create_model():
	Model = Net()
	Model.add(Conv(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=0))
	Model.add(MaxPool(in_channels=32, kernel_size=(2,2), stride=2, padding=0))
	Model.add(ReLU())
	Model.add(Conv(in_channels=32, out_channels=16, kernel_size=(6,6), stride=1, padding=0))
	Model.add(MaxPool(in_channels=16, kernel_size=(3,3), stride=2, padding=0))
	Model.add(ReLU())
	Model.add(Flatten())
	Model.add(Dense(in_features=144, out_features=10))
	# Model.add(Softmax())
	return Model
