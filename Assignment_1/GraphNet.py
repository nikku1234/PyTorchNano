class Net:
    def __init__(self,graphnodes) -> None:
        self.graphnodes = graphnodes
        self.order = []
        # pass

    def add(self,layer):
        self.order.append(layer)

