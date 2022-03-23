
class SoinnEdge:
    def __init__(self,node1,node2,age= 0):
        self.node1 = node1
        self.node2 = node2
        self.age = age
        
    def addToNodes(self):
        n1 = self.node1
        n2 = self.node2
        n1.edges[n2] = self
        n2.edges[n1] = self