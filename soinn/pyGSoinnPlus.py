import hashlib
import math
import numpy as np

from numpy import Inf
from numpy.lib.function_base import median
from sklearn.neighbors import NearestNeighbors
from scipy.stats import median_abs_deviation
from scipy.spatial import distance
from .pySoinnNode import SoinnNode
from .pySoinnPlus import SoinnPlus

memoization = {}

class GSoinnPlus(SoinnPlus):
    def __init__(self, FracPow = 2, limit = Inf, ageMax = 50, fractional = False, **kwargs):
        super().__init__(maxEdgeAge=ageMax, **kwargs)

        self.fracParam = FracPow
        self.limit = limit
        self.maxEdgeAge = ageMax
        self.fracFlag = fractional
    
        self.labels = []
        self.hasSubspace = []
        self.subSpace = []
        self.subLabel = []

        # if not kwargs['node']:
        self.deleteNoiseHandler = self.deleteNoiseNodes_Plus
    
        if self.fracFlag:
            self.distanceHandler = self.fracDist
        else:
            self.distanceHandler = distance.euclidean

        ### Other non-related param
        self.e = 0.00001

    def get_key(self, p_vec, q_vec):
        return hashlib.sha1(p_vec).hexdigest() + hashlib.sha1(q_vec).hexdigest()

    # Minkowski distance
    def fracDist(self, src, target):
        # memoization is used to reduce unnecessary calculations ... makes a BIG difference
        memoize = True
        dist = []
        for tgt in target:
            if memoize:
                key = self.get_key(src, tgt)
                x = memoization.get(key)
                if x is None:
                    diff = src - tgt
                    diff_fraction = abs(diff)**self.fracParam
                    dist.append(max(math.pow(np.sum(diff_fraction), 1/self.fracParam), self.e))
                    #return max(math.pow(np.sum(diff_fraction), 1/self.fracParam), self.e)
                else:
                    #return x
                    dist.append(x)
            else:
                diff = src - tgt
                diff_fraction = diff**self.fracParam
                dist.append(max(math.pow(np.sum(diff_fraction), 1/self.fracParam), self.e))
                #return max(math.pow(np.sum(diff_fraction), 1/self.fracParam), self.e)

        return dist

    def classify(self, X, K):
        neigh = NearestNeighbors(n_neighbors=K)
        nodes_weight = [node.weight for node in self.nodes]
        neigh.fit(nodes_weight)
        dist, index = neigh.kneighbors(X) 

        print(dist)

        labels = []
        # labels check
        for idx in index:
            labels.append([self.labels[i] for i in idx ]) # self.labels[index]

        subspace_neigh = {}
        # Correct index in ghost node
        for i, k_set in enumerate(labels):
            for j, label in enumerate(k_set):
                if self.hasSubspace[index[i][j]]:
                    if index[i][j] not in subspace_neigh:
                        subspace_neigh[index[i][j]] = NearestNeighbors(n_neighbors=1)
                        subspace_neigh[index[i][j]].fit(self.subSpace[index[i][j]])
                    
                    sub_dist, idx = subspace_neigh[index[i][j]].kneighbors([X[i]])
                    dist[i][j] = sub_dist[0][0]

                    labels[i][j] = self.subLabel[index[i][j]][idx[0][0]]

        print(dist)
        print(labels)

        ## Score calculation
        score = np.zeros([len(X), max(self.labels)+1])
        classW = np.zeros([len(X), max(self.labels)+1])

        for i in range(len(X)):
            for k in range(K):
                # Inverse Euclidean distance
                classW[i][labels[i][k]] = classW[i][labels[i][k]] + (1/dist[i][k])

            for c in range(max(self.labels)+1):
                score[i][c] = (classW[i][c] / sum(classW[i]))

        # Classify
        preds = np.argmax(score, axis=1)

        return preds, labels, score

    def findNearestNodes(self, kth, signal):
        indexes = np.zeros(kth, dtype=np.int32)
        sqDists = np.zeros(kth)

        D = self.distanceHandler(signal, self.nodes_weight)
        for i in range(kth):
            indexes[i] = np.argmin(D)
            sqDists[i] = D[indexes[i]]
            D[indexes[i]] = Inf
        
        return indexes, sqDists

    def inputSignal(self, signal, label):
        self.signalNum += 1

        if not label:
            label = 0


        ### Initialization state
        if len(self.nodes) < 3:
            self.addNode(signal, label)
            return

        self.nodes_weight = [node.weight for node in self.nodes]
        winners, dists = self.findNearestNodes(2,signal);
        #dists = dists[0]
        #winners = winners[0]
        simThresholds = self.calculateSimiralityThresholds(winners) # np.squeeze(np.array(self.calculateSimiralityThresholds(winners)))

        run_vars = [x == 0 for x in self.runThVariance]

        if any(run_vars):
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
            self.addNode(signal, label)
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
            
            self.updateWinner(winners[0], signal, label)
            self.updateAdjacentNodes(winners[0], signal)

        if (self.signalNum % self.deleteNodePeriod == 0):
            self.deleteNoiseHandler()
            
    def addNode(self,weight, label):
        # num = len(self.nodes)
        self.nodes.append(SoinnNode(weight, self.signalNum))
        self.winningTimes.append(1)
        self.winTS.append(self.signalNum)
        self.nodeTS.append(self.signalNum)

        self.labels.append(label)
        self.hasSubspace.append(False)
        self.subSpace.append([])
        self.subLabel.append([])

        if self.enableTracking:
            self.trackInput.append([weight])
            self.trackInputIdx.append([self.signalNum])

    def updateWinner(self, winnerIndex, signal, label):
        super().updateWinner(winnerIndex, signal)

        # Check Label
        if self.labels[winnerIndex] != label:
            self.hasSubspace[winnerIndex] = True
            if not self.subSpace[winnerIndex]:
                self.subSpace[winnerIndex].append(self.nodes[winnerIndex].weight)
                self.subLabel[winnerIndex].append(self.labels[winnerIndex])
            
            self.subSpace[winnerIndex].append(signal)
            self.subLabel[winnerIndex].append(label)

    def calculateSimiralityThresholds(self, winners):
        simTh = []
        for winner in winners:
            simTh.append(self.getSimilarityThreshold(self.nodes[winner]))
        
        return simTh

    def getSimilarityThreshold(self,soinnNode):
        if len(soinnNode.edges) > 0:
            maxDist = 0
            for neighbor in soinnNode.edges.keys():
                dis = self.distanceHandler(soinnNode.weight, [neighbor.weight])[0]
                if (dis > maxDist):
                    maxDist = dis
            return maxDist
        else:
            minDist = -1
            for node in self.nodes:
                if (node != soinnNode):
                    dis = self.distanceHandler(soinnNode.weight, [node.weight])[0]
                    if (minDist == -1 or dis < minDist):
                        minDist = dis
            return minDist

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

    def removeUnnecessaryNode(self, winnerIndex):
        removeList = []
        removeIndex = []
        for i, node in enumerate(self.nodes):
            if len(node.edges) < self.minimum_neighbor and i != winnerIndex:
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
        
        self.labels = [label for i, label in enumerate(self.labels) if i not in removeIndex]
        self.hasSubspace = [hasSubspace for i, hasSubspace in enumerate(self.hasSubspace) if i not in removeIndex]
        self.subSpace = [subSpace for i, subSpace in enumerate(self.subSpace) if i not in removeIndex]
        self.subLabel = [subLabel for i, subLabel in enumerate(self.subLabel) if i not in removeIndex]

        new_winners = winnerIndex - sum(removeIndex < winnerIndex)
        return new_winners # len(removeList) > 0

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
            new_labels = []
            new_hasSubspace = []
            new_subSpace = []
            new_subLabel = []
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
                    new_labels.append(self.labels[i])
                    new_hasSubspace.append(self.hasSubspace[i])
                    new_subSpace.append(self.subSpace[i])
                    new_subLabel.append(self.subLabel[i])
            
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

            self.labels = new_labels
            self.hasSubspace = new_hasSubspace
            self.subSpace = new_subSpace
            self.subLabel = new_subLabel