from math import sqrt
from collections import deque
import numpy as np
from soinn.soinn import Soinn
from scipy import stats
from scipy.sparse import dok_matrix, tril
from scipy.sparse.csgraph import breadth_first_order

class NodePlus:
    def __init__(self, signal, timestamp):
        self.original_weight = signal
        self.winning_ts = timestamp # Last winning timestamp
        self.node_ts = timestamp # Created timestamp



class SoinnPlus(Soinn):
    def __init__(self, delete_node_period=300, max_edge_age=50,
                 init_node_num=3):
        
        super().__init__(delete_node_period, max_edge_age, init_node_num)

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

        self.runThVariance = np.array([0, 0], dtype=np.float64)
        self.runThMean = np.array([0, 0], dtype=np.float64)
        self.linkCreated = 0
        self.nodeDeleted = 0
        self.edgeDeleted = 0

        # self.nodes = []
        self.winning_times = []
        self.winning_ts = []
        self.node_ts = []

    def __check_signal(self, signal: np.ndarray):
        """
        check type and dimensionality of an input signal.
        If signal is the first input signal, set the dimension of it as
        self.dim. So, this method have to be called before calling functions
        that use self.dim.
        :param signal: an input signal
        """
        if isinstance(signal, list):
            signal = np.array(signal)
        if not(isinstance(signal, np.ndarray)):
            raise TypeError()
        if len(signal.shape) != 1:
            raise TypeError()
        if self.dim is None:
            self.dim = signal.shape[0]
        else:
            if signal.shape[0] != self.dim:
                raise TypeError()
        return signal

    def __add_node(self, signal: np.ndarray):
        n = self.nodes.shape[0]
        self.nodes.resize((n + 1, self.dim), refcheck=False)
        self.nodes[-1, :] = signal
        self.winning_times.append(1)
        self.adjacent_mat.resize((n + 1, n + 1))

        ## Soinn Plus
        self.winning_ts.append(self.num_signal)
        self.node_ts.append(self.num_signal)
    
    def __find_nearest_nodes(self, num: int, signal: np.ndarray):
        n = self.nodes.shape[0]
        indexes = [0] * num
        sq_dists = [0.0] * num
        D = np.sum((self.nodes - np.array([signal] * n))**2, 1)
        for i in range(num):
            indexes[i] = np.nanargmin(D)
            sq_dists[i] = D[indexes[i]]
            D[indexes[i]] = float('nan')
        return indexes, sq_dists

    def __calculate_similarity_thresholds(self, node_indexes):
        sim_thresholds = []
        for i in node_indexes:
            pals = self.adjacent_mat[i, :]
            if len(pals) == 0:
                idx, sq_dists = self.__find_nearest_nodes(2, self.nodes[i, :])
                sim_thresholds.append(sq_dists[1])
            else:
                pal_indexes = []
                for k in pals.keys():
                    pal_indexes.append(k[1])
                sq_dists = np.sum((self.nodes[pal_indexes] - np.array([self.nodes[i] * len(pal_indexes)]))**2, 1)
                sim_thresholds.append(np.max(sq_dists))
        return sim_thresholds

    def __add_edge(self, node_indexes):
        return self.__set_edge_weight(node_indexes, 1)

    def __increment_edge_ages(self, winner_index):
        for k, v in self.adjacent_mat[winner_index, :].items():
            self.__set_edge_weight((winner_index, k[1]), v + 1)

    def __collect_cluster_edge_age(self, seed):
        queue = deque()
        queue.append(seed)
        edge_list = []
        seen_set = set()

        search_mat = self.adjacent_mat.copy()

        while queue:
            node = queue.popleft()
            if node in seen_set:
                continue

            seen_set.add(node)
            for k, w in self.adjacent_mat[node, :].items():
                if w > 0 and search_mat[node, k[1]]:
                    edge_list.append(w)
                    queue.append(k[1])
                    search_mat[node, k[1]] = 0
                    search_mat[k[1], node] = 0

        return edge_list

    def __delete_old_edges(self, winner_index):
        edge_ages = self.__collect_cluster_edge_age(winner_index)
        q3, q1 = np.percentile(edge_ages, [75, 25])
        th = self.paramEdge*(q3-q1)

        if th == 0:
            return winner_index
        
        curTh = q3 + th
        ratio = self.edgeDeleted / (self.edgeDeleted + len(edge_ages))

        delThreshold = (self.edgeAvgLtDel*ratio) + curTh*(1-ratio)

        candidates = []
        sum_del_edges = 0
        for k, v in self.adjacent_mat[winner_index, :].items():
            if v > delThreshold:
                candidates.append(k[1])
                sum_del_edges += v
                self.__set_edge_weight((winner_index, k[1]), 0)
        
        if candidates:
            self.edgeAvgLtDel = (self.edgeDeleted*self.edgeAvgLtDel + sum_del_edges) / (self.edgeDeleted + len(candidates))
        
        self.edgeDeleted += len(candidates)
        
        delete_indexes = []
        for i in candidates:
            if len(self.adjacent_mat[i, :]) == 0:
                delete_indexes.append(i)

        self.__delete_nodes(delete_indexes)
        delete_count = sum([1 if i < winner_index else 0 for i in delete_indexes])
        return winner_index - delete_count

    def __set_edge_weight(self, index, weight):
        is_new_edge = False
        if self.adjacent_mat[index[0], index[1]] == 0 or self.adjacent_mat[index[1], index[0]] == 0:
            is_new_edge = True
        self.adjacent_mat[index[0], index[1]] = weight
        self.adjacent_mat[index[1], index[0]] = weight

        return is_new_edge

    def __update_winner(self, winner_index, signal):
        self.winning_times[winner_index] += 1
        w = self.nodes[winner_index]
        self.nodes[winner_index] = w + (signal - w)/self.winning_times[winner_index]
        self.winning_ts[winner_index] = self.num_signal

    def __update_adjacent_nodes(self, winner_index, signal):
        pals = self.adjacent_mat[winner_index]
        for k in pals.keys():
            i = k[1]
            w = self.nodes[i]
            self.nodes[i] = w + (signal - w)/(100 * self.winning_times[i])

    def __delete_nodes(self, indexes):
        if not indexes:
            return
        n = len(self.winning_times)
        self.nodes = np.delete(self.nodes, indexes, 0)
        remained_indexes = list(set([i for i in range(n)]) - set(indexes))
        self.winning_times = [self.winning_times[i] for i in remained_indexes]

        ### Soinn Plus
        self.winning_ts = [self.winning_ts[i] for i in remained_indexes] 
        self.node_ts = [self.node_ts[i] for i in remained_indexes]

        self.__delete_nodes_from_adjacent_mat(indexes, n, len(remained_indexes))

    def __delete_nodes_from_adjacent_mat(self, indexes, prev_n, next_n):
        while indexes:
            next_adjacent_mat = dok_matrix((prev_n, prev_n))
            for key1, key2 in self.adjacent_mat.keys():
                if key1 == indexes[0] or key2 == indexes[0]:
                    continue
                if key1 > indexes[0]:
                    new_key1 = key1 - 1
                else:
                    new_key1 = key1
                if key2 > indexes[0]:
                    new_key2 = key2 - 1
                else:
                    new_key2 = key2
                #Because dok_matrix.__getitem__ is slow,
                #access as dictionary.
                next_adjacent_mat[new_key1, new_key2] = super(dok_matrix, self.adjacent_mat).__getitem__((key1, key2))
            self.adjacent_mat = next_adjacent_mat.copy()
            indexes = [i-1 for i in indexes]
            indexes.pop(0)
        self.adjacent_mat.resize((next_n, next_n))

    def __delete_noise_nodes(self):
        data = np.where((sum(self.adjacent_mat ) != 0).toarray())[1]
        noise = [ele for ele in range(self.adjacent_mat.shape[0]) if ele not in data]

        IT = np.subtract(self.num_signal, self.winning_ts) 
        UT = np.divide(IT, self.winning_times)
        ut_data = UT[data]
        med = np.median(ut_data)
        mad = stats.median_abs_deviation(ut_data)
        curTh = med + 2 * mad

        ### Check if any node should be deleted
        ratio = self.nodeDeleted / (self.nodeDeleted + len(data))
        noise_lv = len(noise) / len(self.nodes)
        delThreshold = self.nodeDelTh*ratio + (curTh*(1-ratio)*(1-noise_lv))
        self.curNodeTh = delThreshold



        ### Original section
        n = len(self.winning_times)
        noise_indexes = []
        for i in range(n):
            if len(self.adjacent_mat[i, :]) < self.min_degree and UT[i] > delThreshold:
                noise_indexes.append(i)
        
        if noise_indexes:
            self.nodeAvgIdleDel = (self.nodeDeleted*self.nodeAvgIdleDel + sum(IT[noise_indexes])) / (self.nodeDeleted + len(noise_indexes))
            self.nodeDelTh = (self.nodeDeleted*self.nodeDelTh + sum(UT[noise_indexes])) / (self.nodeDeleted + len(noise_indexes)) 

        self.__delete_nodes(noise_indexes)
        self.nodeDeleted += len(noise_indexes)
    
    def show_network_stats(self):
        print("Number of Nodes:", len(self.nodes))
        print("Running Variance Threshold:", self.runThVariance)
        print("Running Mean:", self.runThMean)
        print("Link Created:", self.linkCreated)
        print("Node Deleted:", self.nodeDeleted)
        print("Edge Deleted:", self.edgeDeleted)


    def fit(self, X):
        """
        train data in batch manner
        :param X: array-like or ndarray
        """
        self._reset_state()
        for x in X:
            self.input_signal(x)
        # self.labels_ = self.__label_samples(X)
        return self

    def input_signal(self, signal: np.ndarray):
        """
        Input a new signal one by one, which means training in online manner.
        fit() calls __init__() before training, which means resetting the
        state. So the function does batch training.
        :param signal: A new input signal
        :return:
        """

        signal = self.__check_signal(signal)
        self.num_signal += 1

        if self.nodes.shape[0] < self.init_node_num:
            self.__add_node(signal)
            return

        winners, dists = self.__find_nearest_nodes(2, signal)
        sim_thresholds = self.__calculate_similarity_thresholds(winners)

        print("Winners:", winners)

        run_vars = [x == 0 for x in self.runThVariance]


        # Check if the network should create the link
        if all(run_vars):
            print("Force Create Link", run_vars)
            eFlag = 1
        else:
            th = self.paramC*np.sqrt(self.runThVariance/self.linkCreated)
            # get data index
            data = np.where((sum(self.adjacent_mat ) != 0).toarray())[1]
            trustLV = np.array(self.winning_times)[winners] / np.array(self.winning_times)[data].max()
            condition = []
            for i in range(len(sim_thresholds)):
                value = sim_thresholds[i] * (1-trustLV[i])
                th1 = self.paramC*sqrt(self.runThVariance[i] / self.linkCreated)
                result = value < self.runThMean[i] + th1
                condition.append(result)
                
            eFlag = any(condition)
            print("Link threshold", th)
            print("trust LV", trustLV)
            ########## Reach here for implementation <<<<<<<<<-----------------
            # trustLv = self.winningTimes(winners) / max(self.winningTimes[data])
            # T = [L[i] for i in Idx] get all winning time in the node
            # winning_times = np.array([self.nodes[idx].winningTimes for idx in winners])
            # max_trust = max([self.nodes[idx].winningTimes for idx in data])
            # trustLv = winning_times / max_trust # max([node.winningTimes for node in self.nodes])
            # link_value = simThresholds * (1-trustLv)
            # threshold = (self.runThMean + th)
            # eFlag = any(link_value < threshold)

        if (dists[0] > sim_thresholds[0] or dists[1] > sim_thresholds[1]): # and not eFlag:
            self.__add_node(signal)
        else:
            ### New Rule 
            # if eFlag allow then add edge
            if eFlag:
                is_new = self.__add_edge(winners)
                if is_new:
                    self.linkCreated += 1
                    pre_mean = self.runThMean

                    # Update mean and var of sim threshold
                    self.runThMean = self.runThMean + (sim_thresholds - self.runThMean) / self.linkCreated
                    self.runThVariance = self.runThVariance + (sim_thresholds - pre_mean) * (sim_thresholds - self.runThMean)

            self.__increment_edge_ages(winners[0])
            winners[0] = self.__delete_old_edges(winners[0])
            self.__update_winner(winners[0], signal)
            self.__update_adjacent_nodes(winners[0], signal)

        if self.num_signal % self.delete_node_period == 0:
            self.__delete_noise_nodes()

        self.show_network_stats()