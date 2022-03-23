import pathlib
import pickle

import numpy as np

from numpy.random import seed
from numpy.random import randint

from soinn.pySoinnPlus import SoinnPlus
from soinn.pyGSoinnPlus import GSoinnPlus

from tensorflow import keras

seed(1)

total_signal = 10000
num_features = 3
signal_1 = randint(0, 100, [total_signal, num_features])
label_1 = randint(0, 2, total_signal)

signal_2 = randint(200, 300, [total_signal, num_features])
label_2 = randint(0, 2, total_signal)

signals = np.concatenate((signal_1, signal_2))
labels = np.concatenate((label_1, label_2))

def train(data, labels):
    soinn = GSoinnPlus(FracPow = 0.2, limit = 10000, ageMax = 50, fractional = True, Lambda = 300, dimension = signals.shape[1])

    for i, sig in enumerate(data):
        # print(i, sig)

        soinn.inputSignal(sig.flatten(), labels[i])

        # if len(soinn.nodes) != len(soinn.winningTimes):

        #     ### Sanity Check
        #     print("Len nodes = {}".format(len(soinn.nodes)))
        #     print("Len nodeTS = {}".format(len(soinn.nodeTS)))
        #     print("Len winTS = {}".format(len(soinn.winTS)))
        #     print("Len winningTimes = {}".format(len(soinn.winningTimes)))
        #     print("Len trackInput = {}".format(len(soinn.trackInput)))
        #     print("Len trackInputIdx = {}".format(len(soinn.trackInputIdx)))
        #     pass

    #print(soinn.nodes)
    print("Total Nodes" ,len(soinn.nodes))
    print("Total Sub Space", sum(soinn.hasSubspace))
    
    all_samples = sum([len(subspace) for subspace in soinn.subSpace])
    print("Total samples in sub space", all_samples)

    return soinn
    query_data = randint(0, 300, [5, num_features])

    print(query_data)

    results = soinn.classify(query_data, 3)
    print(results)
def test(soinn, samples, gt):
    preds, labels, score = soinn.classify(samples, 1)


def run_experiment():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    pickle_data = pathlib.Path('soinn.pickle')

    if pickle_data.exists():
        with open('soinn.pickle', 'rb') as file:
            soinn = pickle.load(file)
    else:
        soinn = train(x_train, y_train)

        with open('soinn.pickle', 'wb') as file:
            pickle.dump(soinn, file)

    test(soinn, x_test)

if __name__ == "__main__":
    run_experiment()