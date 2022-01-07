class Net:
    def __init__(self) -> None:
        # self.graphnodes = graphnodes
        self.order = []
        # pass

    def add(self,layer):
        # print(layer)
        self.order.append(layer)
        # print(self.order)
        # print(len(self.order))s

    def forward(self,value):
        temp = value
        for i in self.order:
            # print(i)
            # print(temp.shape)
            temp = i.forward(temp)
            # print("done",i)
        return temp  

    def backward(self,dx):
        temp = dx
        for i in self.order[::-1]:
            # print("Shape of the gradient returned",temp.shape)
            temp = i.backward(temp)
            # print("done",i)
        return temp
    
    def update(self):
        for i in self.order:
            i.update()

