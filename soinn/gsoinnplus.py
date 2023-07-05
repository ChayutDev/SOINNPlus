import hashlib
from math import sqrt
from collections import deque
import math
import numpy as np
from soinn.soinn import Soinn
from scipy import stats
from scipy.sparse import dok_matrix, tril
from scipy.sparse.csgraph import breadth_first_order
from sklearn.neighbors import NearestNeighbors

from soinn.soinnplus import SoinnPlus

memoization = {}

class GSoinnPlus(SoinnPlus):
    def __init__(self, fracPow = 2, limit = 10000, **kwargs):
        
        super().__init__(**kwargs)

        # GSOINN+ Parameter
        self.fracParam = fracPow
        self.fracFlag = True
        self.limit = limit

        self.labels = []

        # Ghost Node Parameter
        self.subSpace = {}
        self.distanceHandler = None
        

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

    def __add_node(self, signal: np.ndarray, label):
        n = self.nodes.shape[0]
        self.nodes.resize((n + 1, self.dim), refcheck=False)
        self.nodes[-1, :] = signal
        self.winning_times.append(1)
        self.adjacent_mat.resize((n + 1, n + 1))

        ## Soinn Plus
        self.winning_ts.append(self.num_signal)
        self.node_ts.append(self.num_signal)

        ## GSoinnPlus
        self.labels.append(label)
    
    def __find_nearest_nodes(self, num: int, signal: np.ndarray):
        n = self.nodes.shape[0]
        indexes = [0] * num
        sq_dists = [0.0] * num
        # Original SOINN
        # D_old = np.sum((self.nodes - np.array([signal] * n))**2, 1)

        # GSoinnPlus
        D = self.fractional_distance(signal, self.nodes, self.fracParam)
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
                # sq_dists_old = np.sum((self.nodes[pal_indexes] - np.array([self.nodes[i] * len(pal_indexes)]))**2, 1)
                sq_dists = self.fractional_distance(self.nodes[pal_indexes], np.array([self.nodes[i] * len(pal_indexes)]), self.fracParam)
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

        if len(edge_ages) == 0:
            return winner_index
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

    def __update_winner(self, winner_index, signal, label):
        self.winning_times[winner_index] += 1
        w = self.nodes[winner_index]
        self.nodes[winner_index] = w + (signal - w)/self.winning_times[winner_index]
        self.winning_ts[winner_index] = self.num_signal

        ### Label checking for sub space
        if self.labels[winner_index] != label:
            if winner_index not in self.subSpace:
                self.subSpace[winner_index] = {
                    'nodes': [w, signal],
                    'labels': [self.labels[winner_index], label]
                }
                self.labels[winner_index] = -1
            else:
                self.subSpace[winner_index]['nodes'].append(signal)
                self.subSpace[winner_index]['labels'].append(label)

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

        ### G Soinn Plus 
        self.labels = [self.labels[i] for i in remained_indexes]
        for i in indexes:
            if i in self.subSpace:
                del self.subSpace[i]
            

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


    def fit(self, X, labels):
        """
        train data in batch manner
        :param X: array-like or ndarray
        """
        self._reset_state()
        for x, label in zip(X, labels):
            self.input_signal(x, label)
        # self.labels_ = self.__label_samples(X)
        # return self

    def input_signal(self, signal: np.ndarray, label: int = 0):
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
            self.__add_node(signal, label)
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


        if (dists[0] > sim_thresholds[0] or dists[1] > sim_thresholds[1]): # and not eFlag:
            self.__add_node(signal, label)
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
            self.__update_winner(winners[0], signal, label)
            self.__update_adjacent_nodes(winners[0], signal)

        if self.num_signal % self.delete_node_period == 0:
            self.__delete_noise_nodes()

        self.show_network_stats()

    def fractional_distance(self, a, target_list, fraction=2.0):
        results = []
        for b in target_list:
            dist = self._fractional_distance(a, b, fraction)
            results.append(dist)

        return results
        

    def _fractional_distance(self, p_vec, q_vec, fraction=2.0):
        """
        This method implements the fractional distance metric. I have implemented memoization for this method to reduce
        the number of function calls required. The net effect is that the algorithm runs 400% faster. A similar approach
        can be used with any of the above distance metrics as well.
        :param p_vec: vector one
        :param q_vec: vector two
        :param fraction: the fractional distance value (power)
        :return: the fractional distance between vector one and two
        """
        # memoization is used to reduce unnecessary calculations ... makes a BIG difference
        memoize = True
        if memoize:
            key = self.get_key(p_vec, q_vec)
            x = memoization.get(key)
            if x is None:
                diff = p_vec - q_vec
                diff_fraction = diff**fraction
                # return max(math.pow(np.sum(diff_fraction), 1/fraction), self.e)
                return math.pow(np.sum(diff_fraction), 1/fraction)
            else:
                return x
        else:
            diff = p_vec - q_vec
            diff_fraction = diff**fraction
            # return max(math.pow(np.sum(diff_fraction), 1/fraction), self.e)
            return math.pow(np.sum(diff_fraction), 1/fraction)

    def classify(self, X, K):
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(self.nodes)
        dists, indices = nbrs.kneighbors(X)
        labels = self.labels(indices)

        knn_memo = {}

        for i, x in enumerate(X):
            for j in range(K):
                if indices[i,j] in self.subSpace:
                    if indices[i,j] not in knn_memo:
                        knn_memo[indices(i,j)] = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(self.subSpace[indices(i,j)]["nodes"])
                        labels[i,j] = self.subSpace[indices(i,j)]["labels"] 

    @staticmethod    
    def get_key(p_vec, q_vec):
        """
        This method returns a unique hash value for two vectors. The hash value is equal to the concatenated string of
        the hash value for vector one and vector two. E.g. is hash(p_vec) = 1234 and hash(q_vec) = 5678 then get_key(
        p_vec, q_vec) = 12345678. Memoization improved the speed of this algorithm 400%.
        :param p_vec: vector one
        :param q_vec: vector two
        :return: a unique hash
        """
        # return str(hash(tuple(p_vec))) + str(hash(tuple(q_vec)))
        return str(hashlib.sha1(p_vec)) + str(hashlib.sha1(q_vec))