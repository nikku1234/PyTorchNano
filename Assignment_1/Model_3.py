from GraphNet import Net
from Layers_copy import *

def create_model():
	Model = Net()
	Model.add(Conv(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=1))
	Model.add(MaxPool(in_channels=16, kernel_size=(4,4), stride=2, padding=2))
	Model.add(ReLU())
	Model.add(Conv(in_channels=16, out_channels=16, kernel_size=(5,5), stride=1, padding=2))
	Model.add(MaxPool(in_channels=16, kernel_size=(2,2), stride=2, padding=1))
	Model.add(ReLU())
	Model.add(Conv(in_channels=16, out_channels=4, kernel_size=(3,3), stride=1, padding=0))
	Model.add(AvgPool(in_channels=4, kernel_size=(2,2), stride=2, padding=1))
	# Model.add(Sigmoid())
	Model.add(Flatten())
	Model.add(Dense(in_features=64, out_features=10))
	# Model.add(Softmax())
	return Model
