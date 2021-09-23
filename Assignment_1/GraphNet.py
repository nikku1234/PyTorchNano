class Net:
    def __init__(self) -> None:
        # self.graphnodes = graphnodes
        self.order = []
        # pass

    def add(self,layer):
        print(layer)
        self.order.append(layer)
        # print(self.order)
        print(len(self.order))

    def forward(self,value):
        for i in self.order:
            i.forward(value)

    def backward(self,dx):
        print("nikhil")

