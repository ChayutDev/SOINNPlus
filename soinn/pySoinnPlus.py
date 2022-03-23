# -*- coding: utf-8 -*-
'''
This is one layer version of the original soinn. It uses Euclidean distance function for similarity measurment. 

'''
from numpy.lib.function_base import median
from scipy.spatial import distance
from scipy.stats import median_abs_deviation
from sklearn.neighbors import NearestNeighbors

from . import pySoinnNode
#reload(pySoinnNode)
from .pySoinnNode import SoinnNode

from . import pySoinnEdge
#reload(pySoinnEdge)
from .pySoinnEdge import SoinnEdge

import numpy as np

class SoinnPlus:
    def __init__(self,Lambda,maxEdgeAge,dimension, node = True, edge = True, learningRate = 0.5,minimum_neighbor = 1):
        """
        Legacy Options: 
            'lambda' (default: 300)
                A period deleting nodes. The nodes that doesn't satisfy
                some condition are deleted every this period.
            'ageMax' (default: 50)
                The maximum of edges' ages. If an edge's age is more
                than this, the edge is deleted.
            'dim' (default: 2)
                signal's dimension
            
            Plus Options:
            'node' (default: 1)
                1 - Enable plus version of node deletion
                0 - Disable plus version of node deletion
            'edge' (default: 1)
                1 - Enable plus version of node linking
                0 - Disable plus version of node linking
        """
        
        # Internal attribute
        self.nodeDelTh = 0
        self.nodeAvgIdleDel = 0
        self.edgeAvgLtDel = 0
        self.curNodeTh = None
        self.curEdgeTh = None
        self.enableTracking = 1
        self.minDegree = 1
        self.paramEdge = 2
        self.paramC = 2
        self.paramAlpha = 2

        # Parameter
        self.dimension = dimension
        self.deleteNodePeriod = Lambda
        self.maxEdgeAge = maxEdgeAge

        # SOINN+ Option
        self.nodeFlag = node
        self.edgeFlag = edge

        # Data related attr
        self.signalNum = 0
        self.nodes = []
        self.trackInput = []
        self.trackInputIdx = []
        self.winningTimes = []
        self.winTS = []
        self.nodeTS = []
        self.adjacencyMat = []
        self.linkCreated = 0
        self.nodeDeleted = 0
        self.edgeDeleted = 0
        self.runThVariance = np.array([0, 0])
        self.runThMean = np.array([0, 0])

        if not node:
            self.deleteNoiseHandler = self.deleteNoiseNodes_Original
        else:
            self.deleteNoiseHandler = self.deleteNoiseNodes_Plus
            self.deleteNodePeriod = 1

        if not edge:
            self.deleteEdgeHandler = self.deleteOldEdges_Original
        else:
            self.deleteEdgeHandler = self.deleteOldEdges_Plus
            self.deleteNodePeriod = 1

        # self.nodeTh = Lambda
        # self.edgeTh = maxEdgeAge
        # self.learningRate = learningRate
        
        # self.edges = []
        self.NNModel = None
        self.currentWinners = []
        self.minimum_neighbor = minimum_neighbor
        
    def inputSignal(self,signal):
        if len(signal) != self.dimension:
            return -1
        
        self.signalNum += 1
        if len(self.nodes) < 3:
            self.addNode(signal, 0)
            return 0
        
		# Find nesrest neighbor
        data = [node.weight for node in self.nodes]
        self.NNModel = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data)
        dists, winners = self.NNModel.kneighbors(np.array(signal).reshape(1, -1)) # What if there are multiple node that share the same distance
        dists = dists[0]
        winners = winners[0]
        simThresholds = self.calculateSimiralityThresholds(winners)
        
        run_vars = [x == 0 for x in self.runThVariance]

        if all(run_vars):
            eFlag = True
            isNew = self.addEdge(self.nodes[winners[0]], self.nodes[winners[1]])
            if isNew:
                self.linkCreated = self.linkCreated + 1
                preMean = self.runThMean
                    
                # Update the mean and variance of similarity threshold
                self.runThMean = self.runThMean + (simThresholds - self.runThMean) / self.linkCreated
                self.runThVariance = self.runThVariance + (simThresholds - preMean) * (simThresholds - self.runThMean)
        else:
            th = self.paramC*np.sqrt(self.runThVariance/self.linkCreated)
            noises, data = self.getNoiseNode()
            # trustLv = self.winningTimes(winners) / max(self.winningTimes[data])
            # T = [L[i] for i in Idx] get all winning time in the node
            winning_times = np.array([self.nodes[idx].winningTimes for idx in winners])
            max_trust = max([self.nodes[idx].winningTimes for idx in data])
            trustLv = winning_times / max_trust # max([node.winningTimes for node in self.nodes])
            link_value = simThresholds * (1-trustLv)
            threshold = (self.runThMean + th)
            eFlag = any(link_value < threshold)


        if any(dists > simThresholds):
            self.addNode(signal, 0)
        else:
            if eFlag:
                isNew = self.addEdge(self.nodes[winners[0]], self.nodes[winners[1]])
                if isNew:
                    self.linkCreated = self.linkCreated + 1
                    preMean = self.runThMean
                    # Update the mean and variance of similarity threshold
                    self.runThMean = self.runThMean + (simThresholds - self.runThMean) / self.linkCreated
                    self.runThVariance = self.runThVariance + (simThresholds - preMean) * (simThresholds - self.runThMean)

            self.nodes[winners[0]].incrementEdgeAges()
            # self.incrementEdgeAges(winners[0])
            winners[0] = self.deleteEdgeHandler(winners[0])
            
            self.updateWinner(winners[0], signal)
            self.updateAdjacentNodes(winners[0], signal)

        if (self.signalNum % self.deleteNodePeriod == 0):
            self.deleteNoiseHandler()
            # self.removeUnnecessaryNode()

        #winners = self.findWinnerAndSecondWinner(signal)
		
		
        #if (self.isWithinThreshold(winners,signal)): 
        #    self.addEdge(winners[0],winners[1])
        #    winners[0].increaseEdgesAge()
        #    winners[0].learningTime += 1
        #    self.moveNode(winners[0],signal)
        #    self.removeDeadEdge(winners[0])
        #else:
        #    self.addNode(signal)
             
        return
    
    def addNode(self,weight, *args):
        # num = len(self.nodes)
        self.nodes.append(SoinnNode(weight, self.signalNum))
        self.winningTimes.append(1)
        self.winTS.append(self.signalNum)
        self.nodeTS.append(self.signalNum)

        if self.enableTracking:
            self.trackInput.append([weight])
            self.trackInputIdx.append([self.signalNum])

    def addEdge(self,node1,node2):
        if node1 in node2.edges.keys():
            edge = node1.edges[node2]
            edge.age = 0
            return 0
        else:
            edge = SoinnEdge(node1,node2)
            node1.edges[node2] = edge
            node2.edges[node1] = edge      
            return 1

    def updateWinner(self, winnerIndex, signal):
        """
        winnerIndex: The index of winner
        signal: inputted new signal vector
        """
        try:
            self.winningTimes[winnerIndex] = self.winningTimes[winnerIndex] + 1
        except:
            print("Some Erorr")
        self.nodes[winnerIndex].winningTimes += 1
        w = self.nodes[winnerIndex].weight
        self.nodes[winnerIndex].weight = w + (signal - w) / self.winningTimes[winnerIndex]
        self.winTS[winnerIndex] = self.signalNum
        self.nodes[winnerIndex].winTS = self.signalNum

        if self.enableTracking:
            self.trackInput[winnerIndex].append(signal)
            self.trackInputIdx[winnerIndex].append(self.signalNum)

    def updateAdjacentNodes(self, winnerIndex, signal):
        pals = []
        for neighbor in self.nodes[winnerIndex].edges.keys():
            neighbor.weight = neighbor.weight + (signal - neighbor.weight) / (100*self.winningTimes[winnerIndex])

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
        
    def calculateSimiralityThresholds(self, winners):
        simTh = []
        for winner in winners:
            simTh.append(self.getSimilarityThreshold(self.nodes[winner]))
        
        return simTh

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
    
    def moveNode(self,node,signal):
        movingRate = float(self.learningRate) / node.learningTime
        neighborMovingRate = movingRate / 100 #This is in the original java code. The reason to why we use 100 is unknown. 
        
        for i in range(0,self.dimension):
            node.weight[i] += movingRate * (signal[i] - node.weight[i])
        
        for neighbor in node.edges.keys():
            for i in range(0,self.dimension):
                neighbor.weight[i] += neighborMovingRate * (signal[i] - neighbor.weight[i])
        
        return
    
    def removeDeadEdge(self, node, threshold):
        isRemove = False
        removed_age = []
        edge_key_list = list(node.edges.keys())
        for nei in edge_key_list:
            edge = node.edges[nei]
            if edge.age > threshold:
                del node.edges[nei]
                del nei.edges[node]
                removed_age.append(edge.age)
                edge.age = -1
                isRemove = True
                
        return isRemove, removed_age
    
    def deleteNoiseNodes_Original(self):
        removeList = []
        for node in self.nodes:
            if len(node.edges) < self.minimum_neighbor:
                removeList.append(node)
        
        for node in removeList:
            self.nodes.remove(node)
            for nei in node.edges.keys():
                del node.edges[nei]
                del nei.edges[node]

    def deleteNoiseNodes_Plus(self):
        noises = []
        data = []
        for i, node in enumerate(self.nodes):
            if len(node.edges) < self.minimum_neighbor:
                noises.append(i)
            else:
                data.append(i)
        
        IT = self.signalNum - np.array(self.winTS)
        UT = IT / self.winningTimes

        mad = median_abs_deviation(UT[data])
        th = self.paramAlpha*mad
        curTh = median(UT[data]) + th

        # Check if there are any node should be deleted
        ratio = self.nodeDeleted / (self.nodeDeleted + len(data))
        noiseLv = len(noises) / len(self.nodes)
        delThreshold = self.nodeDelTh*ratio + curTh*(1-ratio)*(1-noiseLv)
        inactiveIdx = UT > delThreshold
        self.curNodeTh = delThreshold
        inactiveIdx = [i for i, x in enumerate(inactiveIdx) if x]
        
        noiseIdx = list(set(inactiveIdx).intersection(noises))
        
        if noiseIdx:
            self.nodeAvgIdleDel = (self.nodeDeleted * self.nodeAvgIdleDel + sum(IT[noiseIdx])) / (self.nodeDeleted + len(noiseIdx))
            self.nodeDelTh = (self.nodeDeleted * self.nodeDelTh + sum(UT[noiseIdx])) / (self.nodeDeleted + len(noiseIdx))

            # Remove node
            remove_node_list = []
            keep_node_list = []

            new_nodeTS = []
            new_winTS = []
            new_winningTimes = []
            new_trackInput = []
            new_trackInputIdx = []
            # for idx in noiseIdx:
            for i, node in enumerate(self.nodes):
                if i in noiseIdx:
                    remove_node_list.append(self.nodes[i])
                else:
                    keep_node_list.append(self.nodes[i])
                    new_nodeTS.append(self.nodeTS[i])
                    new_winTS.append(self.winTS[i])
                    new_winningTimes.append(self.winningTimes[i])
                    new_trackInput.append(self.trackInput[i])
                    new_trackInputIdx.append(self.trackInputIdx[i])
            
                #node = self.nodes.pop(idx)
            for node in remove_node_list:
                for nei in node.edges.keys():
                    del node.edges[nei]
                    del nei.edges[node]
                
            self.nodes = keep_node_list
            self.nodeTS = new_nodeTS
            self.winTS = new_winTS
            self.winningTimes = new_winningTimes
            self.trackInput = new_trackInput
            self.trackInputIdx = new_trackInputIdx


    
    def deleteNodes(self, indexes):
        self.nodes = [node for i, node in enumerate(self.nodes) if i in indexes]

    def deleteOldEdges_Original(self):
        pass
    
    def deleteOldEdges_Plus(self, winnerIndex):
        edgeAge = self.collectClusterEdgeAge(winnerIndex)
        edges = self.nodes[winnerIndex].edges

        c = np.percentile(edgeAge, 75)
        q3, q1 = np.percentile(edgeAge, [75 ,25])
        iqr = q3 - q1
        th = self.paramEdge*iqr

        if th == 0:
            return winnerIndex
        
        curTh = c + th
        ratio = self.edgeDeleted / (self.edgeDeleted + len(edgeAge))

        # Check if there are any edges to be deleted
        delThreshold = (self.edgeAvgLtDel*ratio + curTh*(1-ratio))
        self.curEdgeTh = delThreshold
        # Try to delete edge and remove unlinked node
        #delFlag = [e.age > delThreshold for e in edges ]
        isRemove, removed_age = self.removeDeadEdge(self.nodes[winnerIndex], delThreshold)
                

        # Update average lifetime of deleted edges
        if isRemove:
            self.edgeAvgLtDel = (self.edgeDeleted*self.edgeAvgLtDel + sum(removed_age)) / (self.edgeDeleted + len(removed_age))

        self.edgeDeleted = self.edgeDeleted + len(removed_age)
        # deletedNodeIndexes = []
        # for node in self.nodes:
        #     if not node.edges

        return self.removeUnnecessaryNode(winnerIndex)


    def collectClusterEdgeAge(self, seed):
        nodeList = [self.nodes[seed]]
        edgeList = set()

        seenNode = set()

        while nodeList:
            node = nodeList.pop()

            if node in seenNode:
                continue
            else:
                seenNode.add(node)
            

            for neighbor in node.edges.keys():
                nodeList.append(neighbor)

                #if not node.edges[neighbor] in edgeList:
                edgeList.add(node.edges[neighbor])

        edgeAge = [edge.age for edge in edgeList]

        return edgeAge
        

    def getNoiseNode(self):
        noiseList = []
        dataList = []
        for i, node in enumerate(self.nodes):
            if len(node.edges) < self.minimum_neighbor:
                noiseList.append(i)
            else:
                dataList.append(i)
        
        return noiseList, dataList

    def removeUnnecessaryNode(self, winnerIndex):
        removeList = []
        removeIndex = []
        for i, node in enumerate(self.nodes):
            if len(node.edges) < self.minimum_neighbor:
                removeList.append(node) 
                removeIndex.append(i)
        
        for node in removeList:
            self.nodes.remove(node)
            for nei in node.edges.keys():
                del node.edges[nei]
                del nei.edges[node]
            
        self.winTS =  [winTS for i, winTS in enumerate(self.winTS) if i not in removeIndex]
        self.nodeTS =  [nodeTS for i, nodeTS in enumerate(self.nodeTS) if i not in removeIndex]
        self.winningTimes =  [winningTimes for i, winningTimes in enumerate(self.winningTimes) if i not in removeIndex]
        self.trackInput =  [trackInput for i, trackInput in enumerate(self.trackInput) if i not in removeIndex]
        self.trackInputIdx =  [trackInputIdx for i, trackInputIdx in enumerate(self.trackInputIdx) if i not in removeIndex]
        
        new_winners = winnerIndex - sum(removeIndex < winnerIndex)
        return new_winners # len(removeList) > 0
            
	
        
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
        
        