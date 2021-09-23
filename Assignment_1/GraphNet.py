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
        temp = value
        for i in self.order:
            temp = i.forward(temp)
            

    def backward(self,dx):
        temp = dx
        for i in self.order[::-1]:
            temp = i.backward(temp)

