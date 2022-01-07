from GraphNet import Net
from Layers_copy import *

def create_model():
	Model = Net()
	Model.add(Conv(in_channels=1, out_channels=32, kernel_size=(7,7), stride=2, padding=0))
	Model.add(AvgPool(in_channels=32, kernel_size=(3,3), stride=2, padding=0),)
	Model.add(Conv(in_channels=32, out_channels=8, kernel_size=(3,3), stride=2, padding=0))
	Model.add(Flatten())
	Model.add(ReLU())
	Model.add(Dense(in_features=32, out_features=48))
	Model.add(Sigmoid())
	Model.add(Dense(in_features=48, out_features=10))
	# Model.add(Softmax())
	return Model
