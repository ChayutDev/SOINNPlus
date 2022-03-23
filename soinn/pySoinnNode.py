# -*- coding: utf-8 -*-
class SoinnNode:
    def __init__(self,weight,signalNum,winningTimes = 1, edges = -1, cluster = 0):
        self.weight = weight
        self.winningTimes = winningTimes
        self.winTS = signalNum
        self.cluster = cluster
        self.edges = edges
        if (edges == -1):
            self.edges = {}
        return
    
    def incrementEdgeAges(self):
        for edge in self.edges.values():
            edge.age += 1
        return
    
    