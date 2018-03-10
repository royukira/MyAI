from Q_learning.Q_Brain_Simply import linear_Q
import numpy as np
import threading
import time


def collect(n_state, threadID=None):

    print("State {0} Tread is starting...\n".format(n_state))

    ACTIONS = [1, 2]  # available actions 1:right, 2:wrong


    train_model = linear_Q(numState=n_state, ActionSet=ACTIONS, greedy=None, learnRate=None,
                           discountFactor=None,
                           Max_episode=None)
    memory, memory_count = train_model.collect_experience(1000*n_state,threadID)
    np.save("./blind_cliffwalk_experience/memory_{0}.npy".format(n_state-1), memory)



class CollectThread (threading.Thread):
    def __init__(self, threadID, n):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.n = n

    def run(self):

        print("开始线程：state {0}".format(self.n-1))
        collect(self.n, self.threadID)
        print("退出线程：state {0}".format(self.n-1))


if __name__ == "__main__":

    N_STATES = [6, 11, 16, 21, 26]

    thread5 = CollectThread("State {0} Thread".format(5), 6)
    thread10 = CollectThread("State {0} Thread".format(10), 11)
    thread15 = CollectThread("State {0} Thread".format(15), 16)
    thread20 = CollectThread("State {0} Thread".format(20), 21)
    thread25 = CollectThread("State {0} Thread".format(25), 26)

    thread5.start()
    thread10.start()
    thread15.start()
    thread20.start()
    thread25.start()

    thread5.join()
    thread10.join()
    thread15.join()
    thread20.join()
    thread25.join()

    print("ALL DONE....")



