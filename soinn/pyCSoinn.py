# -*- coding: utf-8 -*-
'''
This is one layer version of the original soinn. It uses Euclidean distance function for similarity measurment. 

'''
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

import numpy as np
import pandas as pd

import pySoinnNode
reload(pySoinnNode)
from pySoinnNode import SoinnNode

import pySoinnEdge
reload(pySoinnEdge)
from pySoinnEdge import SoinnEdge

class SOINN:
    def __init__(self,dimension,Lambda,maxEdgeAge,learningRate = 0.5,minimum_neighbor = 1):
        self.dimension = dimension
        self.nodeTh = Lambda
        self.edgeTh = maxEdgeAge
        self.learningRate = learningRate
        self.nodes = np.empty([0, dimension])
        self.WT = np.empty([0, 1])
        self.runALR = np.empty([0, 1])
        self.edges = np.empty([0, 0])
        self.NNModel = None;
        self.curWinners = []
        self.curDistances = []
        self.win1NN = []
        self.win1Dist = []
        self.win2NN = []
        self.win2Dist = []
        self.minimum_neighbor = minimum_neighbor
        self.inputNum = 0
        return
    
    def addNode(self,weight):
        self.nodes = np.append(self.nodes, weight)
        nSize = self.nodes.shape
        self.runALR = np.append(self.runALR, 0)
        self.WT = np.append(self.WT, 0)
        self.edge = np.append(self.edge, np.zeros(nSize[0]), 0)
        self.edge = np.append(self.edge, np.zeros(nSize[0]+1, 1), 1)
        return
    
    def findWinnerAndSecondWinner(self,weight): #Modify to 3
        winners = [-1,-1] 
        dwinners = [-1,-1]
        for node in self.nodes:
            d = distance.euclidean(weight,node.weight)     
			if (dwinners[0] == -1 or dwinners[0] > d):
                dwinners[2] = dwinners[1]
				dwinners[1] = dwinners[0]
                dwinners[0] = d
                winners[2] = winners[1]
				winners[1] = winners[0]
                winners[0] = node			
            if (dwinners[1] == -1 or dwinners[1] > d):
                dwinners[2] = dwinners[1]
                dwinners[1] = d
                winners[2] = winners[1]
                winners[1] = node
            elif (dwinners[2] == -1 or dwinners[2] > d):
                dwinners[2] = d
                winners[2] = node
                
        return winners, dwinners
        
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
		self.edge[node1][node2] = 1
		self.edge[node2][node1] = 1
		return
			
	def increaseEdgeAge(self,winner)
		self.edge[winner].where(self.edge[winner] > 0)
    
    def moveNode(self,node,signal):
        #movingRate = float(self.learningRate) / node.learningTime
        movingRate = 1 / self.WT[node]
		#neighborMovingRate = movingRate / 100 #This is in the original java code. The reason to why we use 100 is unknown. 
        
        for i in range(0,self.dimension):
            node.weight[i] += movingRate * (signal[i] - self.nodes[node][i])
        
        #for neighbor in node.edges.keys():
        #    for i in range(0,self.dimension):
        #        neighbor.weight[i] += neighborMovingRate * (signal[i] - neighbor.weight[i])
        
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
		self.curDistances, self.curWinners = self.NNModel.kneighbors(signal)
		self.win1Dist, self.win1NN = self.NNModel.kneighbors(self.curWinners[0])
		self.win2Dist, self.win2NN = self.NNModel.kneighbors(self.curWinners[1])
        #self.curWinners, self.curDistances = self.findWinnerAndSecondWinner(signal)

		sigALR = ALRcalc(signal, self.curWinners)
		win1ALR = ALRcalc(self.nodes[self.curWinners[0]], self.win1NN)
		win2ALR = ALRcalc(self.nodes[self.curWinners[1]], self.win2NN)
		T = max(win1ALR, win2ALR)
		self.runALR[self.curWinners[0]] = self.runALR[self.curWinners[0]] + (sigALR - self.runALR[self.curWinners[0]])/self.WT[self.curWinners[0]]
		dALR = max(0,T-self.runALR[self.curWinners[0]])
		ratio = dALR / T
		Threshold = T*(1-ratio) + self.runALR[self.curWinners[0]]*ratio
		
        #if (self.isWithinThreshold(winners,signal)): 
        if sigALR < Threshold
			self.addEdge(self.curWinners[0],self.curWinners[1])
            self.WT[self.curWinners[0]]+=1
			idx = np.where(self.edge[self.curWinners[0]] > 0)
			for i in idx
				self.edge[self.curWinners[0][i]]+=1
			idx = np.where(self.edge[:][self.curWinners[0]] > 0)
			for i in idx
				self.edge[i][self.curWinners[0]]+=1
			#winners[0].increaseEdgesAge()
            #winners[0].learningTime += 1
            self.moveNode(self.curWinners[0],signal)
            self.removeDeadEdge(self.curWinners[0])
        else:
            self.addNode(signal)
       
        #if (self.inputNum % self.Lambda == 0):
        #    self.removeUnnecessaryNode()
             
        return
		
	def edgeGather(self, winner, lv):
		nbChk = np.zeros(self.nodes.shape, dtype=bool)
		edgeList = []
		nbList = self.edges[winner,:] > 0
		q = deque()
		q.append(winner)
		
		for e in nbList
			edgeList.append([winner, e])
			q.append(e)
			nbChk[winner,e] = true
			nbChk[e,winner] = true
			
		for l in range(lv)
			tmp = deque()
			while q
				node = q.pop()
				nbList = self.edges[node:] > 0
				for e in nbList
					if ~nbChk[node,e]
						edgeList.append([node,e])
						tmp.append(e)
						nbChk[node,e] = true
						nbChk[e,node] = true
				
						
	
	def getRDist(self, A, B, K)
		d = distance.euclidean(A, B) 
		return max(d, self.curDistances[K-1])
	
	def ALRcalc(self, A, B, K) # A: Vector | B: node index
		sumD = 0
		for n in B:
			sumD = sumD + getRDist(A, self.nodes[n], K)
		return sumD/B.size
        
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
        
        