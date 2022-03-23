from numpy.random import seed
from numpy.random import randint

from pySoinnPlus import SoinnPlus
seed(1)

signals = randint(0, 100, [10000, 70])

if __name__ == "__main__":
    soinn = SoinnPlus(300, 100, signals.shape[1])

    for i, sig in enumerate(signals):
        # print(i, sig)

        soinn.inputSignal(sig)

        if len(soinn.nodes) != len(soinn.winningTimes):

            ### Sanity Check
            print("Len nodes = {}".format(len(soinn.nodes)))
            print("Len nodeTS = {}".format(len(soinn.nodeTS)))
            print("Len winTS = {}".format(len(soinn.winTS)))
            print("Len winningTimes = {}".format(len(soinn.winningTimes)))
            print("Len trackInput = {}".format(len(soinn.trackInput)))
            print("Len trackInputIdx = {}".format(len(soinn.trackInputIdx)))
            pass

    #print(soinn.nodes)
    print(len(soinn.nodes))