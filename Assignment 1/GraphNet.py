class GraphNet:
    def __init__(self,graphnodes) -> None:
        self.graphnodes = graphnodes
        self.order = []
        # pass
    
    # def forward(self):
    #     return None

    # def backward(self):
    #     return None

    def add(self,layer):
        self.order.append(layer)

