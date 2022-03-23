# -*- coding: utf-8 -*-
'''
This is one layer version of the original soinn. It uses Euclidean distance function for similarity measurment. 

'''
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

import pySoinnNode
#reload(pySoinnNode)
from pySoinnNode import SoinnNode

import pySoinnEdge
#reload(pySoinnEdge)
from pySoinnEdge import SoinnEdge

class SOINN:
    def __init__(self,dimension,Lambda,maxEdgeAge,learningRate = 0.5,minimum_neighbor = 1):
        self.dimension = dimension
        self.nodeTh = Lambda
        self.edgeTh = maxEdgeAge
        self.learningRate = learningRate
        self.nodes = []
        self.edges = []
        self.NNModel = None
        self.currentWinners = []
        self.minimum_neighbor = minimum_neighbor
        self.inputNum = 0
        return
    
    def addNode(self,weight):
        self.nodes.append(SoinnNode(weight))
        return
    
    def findWinnerAndSecondWinner(self,weight):
        winners = [-1,-1] 
        dwinners = [-1,-1]
        for node in self.nodes:
            d = distance.euclidean(weight,node.weight)            
            if (dwinners[0] == -1 or dwinners[0] > d):
                dwinners[1] = dwinners[0]
                dwinners[0] = d
                winners[1] = winners[0]
                winners[0] = node
            elif (dwinners[1] == -1 or dwinners[1] > d):
                dwinners[1] = d
                winners[1] = node
                
        return winners
        
    def getSimilarityThreshold(self,soinnNode):
        if len(soinnNode.edges) > 0:
            maxDist = 0
            for neighbor in soinnNode.edges.keys():
                dis = distance.euclidean(soinnNode.weight,neighbor.weight) 
                if (dis > maxDist):
                    maxDist = dis
            return maxDist
        else:
            minDist = -1
            for node in self.nodes:
                if (node != soinnNode):
                    dis = distance.euclidean(node.weight,soinnNode.weight)
                    if (minDist == -1 or dis < minDist):
                        minDist = dis
            return minDist

            
    
    def isWithinThreshold(self,winners,weight):
        for winner in winners:
            if distance.euclidean(weight,winner.weight) > self.getSimilarityThreshold(winner):
                return False
        return True
        
    def addEdge(self,node1,node2):
        if (node1 in node2.edges.keys()):
            edge = node1.edges[node2]
            edge.age = 0
            return 0
        else:
            edge = SoinnEdge(node1,node2)
            node1.edges[node2] = edge
            node2.edges[node1] = edge      
            return 1
    
    def moveNode(self,node,signal):
        movingRate = float(self.learningRate) / node.learningTime
        neighborMovingRate = movingRate / 100 #This is in the original java code. The reason to why we use 100 is unknown. 
        
        for i in range(0,self.dimension):
            node.weight[i] += movingRate * (signal[i] - node.weight[i])
        
        for neighbor in node.edges.keys():
            for i in range(0,self.dimension):
                neighbor.weight[i] += neighborMovingRate * (signal[i] - neighbor.weight[i])
        
        return
    
    def removeDeadEdge(self,node):
        isRemove = False
        for nei in node.edges.keys():
            edge = node.edges[nei]
            if edge.age > self.maxEdgeAge:
                del node.edges[nei]
                del nei.edges[node]
                edge.age = -1
                isRemove = True
        return isRemove
    
    def removeUnnecessaryNode(self):
        removeList = []
        for node in self.nodes:
            if len(node.edges) < self.minimum_neighbor:
                removeList.append(node) 
        
        for node in removeList:
            self.nodes.remove(node)
            for nei in node.edges.keys():
                del node.edges[nei]
                del nei.edges[node]
                
        return len(removeList) > 0
            
        
    def inputSignal(self,signal):
        if len(signal) != self.dimension:
            return -1
        
        self.inputNum += 1
        if len(self.nodes) < 2:
            self.addNode(signal)
            return 0
        
		# Find nesrest neighbor
        self.NNModel = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(self.nodes)
        self.currentWinners = self.NNModel.kneighbors(signal)
        #winners = self.findWinnerAndSecondWinner(signal)
		
		
		
        #if (self.isWithinThreshold(winners,signal)): 
        #    self.addEdge(winners[0],winners[1])
        #    winners[0].increaseEdgesAge()
        #    winners[0].learningTime += 1
        #    self.moveNode(winners[0],signal)
        #    self.removeDeadEdge(winners[0])
        #else:
        #    self.addNode(signal)
       
        #if (self.inputNum % self.Lambda == 0):
        #    self.removeUnnecessaryNode()
             
        return
		
	def getRDist
	
        
    def saveToFile(self,filePath):
        f = open(filePath,'w')
        f.write(str(self.dimension)+',')
        f.write(str(self.Lambda )+',')
        f.write(str(self.maxEdgeAge)+',')
        f.write(str(self.learningRate)+',')
        f.write(str(self.inputNum)+'\n')
        nodeNum = len(self.nodes)
        f.write(str(nodeNum)+'\n')
        node2num = {}
        for i in range(nodeNum):
            node = self.nodes[i]
            node2num[node] = i # will be used in edge recording step
            f.write(str(node.learningTime)+','+str(node.cluster)+':')
            f.write(str(node.weight[0]))
            for j in range(1,self.dimension):
                f.write(','+str(node.weight[j]))
            f.write('\n')
        
        edgeNum = len(self.edges)
        f.write(str(edgeNum)+'\n')
        for i in range(edgeNum):
            edge = self.edges[i]  
            n1 = node2num[edge.node1]
            n2 = node2num[edge.node2]
            age = edge.age
            f.write(str(n1)+','+str(n2)+','+str(age)+'\n')
        f.close()
    
    def loadFromFile(self,filePath):
        f = open(filePath,'r')
        tokens = f.readline().strip().split(',')
        self.dimension = int(tokens[0])
        self.Lambda = int(tokens[1])
        self.maxEdgeAge = int(tokens[2])
        self.learningRate = float(tokens[3])
        self.inputNum = int(tokens[4])
        nodeNum = int(f.readline().strip())
        num2node = {} # will be used in edge lording step
        for i in range(nodeNum):
            tmps = f.readline().strip().split(':')
            tokens = tmps[0].split(',')
            learningTime = int(tokens[0])
            cluster = int(tokens[1])
            tokens = tmps[1].split(',')
            weight = []
            for tok in tokens:
                weight.append(float(tok))
            node = SoinnNode(weight,learningTime = learningTime, cluster = cluster)
            num2node[i] = node
            self.nodes.append(node)
        
        edgeNum = int(f.readline().strip())
        for i in range(edgeNum):
            tokens = f.readline().strip().split(',')
            n1 = num2node[int(tokens[0])]
            n2 = num2node[int(tokens[1])]
            age = int(tokens[2])
            edge = SoinnEdge(n1,n2,age)
            edge.addToNodes()
        f.close()
        
        