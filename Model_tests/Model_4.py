from GraphNet import Net
from Layers import *

def create_model():
	Model = Net()
	Model.add(Conv(in_channels=1, out_channels=16, kernel_size=(5,3), stride=1, padding=0))
	Model.add(ReLU())
	Model.add(Conv(in_channels=16, out_channels=8, kernel_size=(1,1), stride=1, padding=0))
	Model.add(AvgPool(in_channels=8, kernel_size=(2,2), stride=2, padding=0))
	Model.add(ReLU())
	Model.add(Conv(in_channels=8, out_channels=8, kernel_size=(3,7), stride=1, padding=1))
	Model.add(MaxPool(in_channels=8, kernel_size=(3,3), stride=3, padding=0),)
	Model.add(Sigmoid())
	Model.add(Conv(in_channels=8, out_channels=16, kernel_size=(1,1), stride=1, padding=0))
	Model.add(AvgPool(in_channels=16, kernel_size=(4,3), stride=3, padding=0))
	Model.add(Flatten())
	Model.add(Dense(in_features=16, out_features=10))
	# Model.add(Softmax())
	return Model
