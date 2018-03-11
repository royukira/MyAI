from Q_learning.Q_Brain_Simply import linear_Q
from Q_learning.Q_Brain_Simply import QBrainSimply
from Q_learning.Q_Brain_Simply import oracle_Q
import numpy as np
import threading


def train(mode,threadID=None):
    """Setting"""
    N_STATES = 16 # the length of the 1 dimensional world
    ACTIONS = [1, 2]  # available actions 1:right, 2:wrong
    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 1 - (1 / N_STATES - 1)  # discount factor
    MAX_EPISODES = 100  # maximum episodes
    batchSize = 10
    max_epoch = 1000

    """
    Load the memory
    """
    memory = np.load("./blind_cliffwalk_experience/memory_15.npy")

    if mode == "LFA":
        train_model = linear_Q(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                               discountFactor=GAMMA,
                               Max_episode=MAX_EPISODES)

        train_w = train_model.batch_linear_train(memory=memory, batchSize=batchSize, max_epoch=max_epoch)

        print("--> Thread: {0} well trained...\n--> {0} Start to test...".format(threadID))

        train_model.test_policy(train_w, 15)

    if mode == "table":
        train = QBrainSimply(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                             discountFactor=GAMMA,
                             Max_episode=MAX_EPISODES)

        train_table = train.batch_table_train(memory, batchSize, max_epoch)

        print("--> Thread: {0} well trained...\n--> {0} Start to test...".format(threadID))

        train.test_policy(train_table, 15)

    if mode == "oracle":
        tableName = "PRIORITY{0}".format(N_STATES-1)

        train = oracle_Q(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                             discountFactor=GAMMA,
                             Max_episode=MAX_EPISODES)

        train.createPriorityTable(tableName)
        train.insertMemory(memory, tableName)
        train_w = train.oracle_train(batchSize, max_epoch, tableName)

        print("--> Thread: {0} well trained...\n--> {0} Start to test...".format(threadID))

        train.test_policy(train_w, 15)

class TrainThread (threading.Thread):
    def __init__(self, threadID, mode):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.mode = mode

    def run(self):

        print("开始线程：state {0}".format(self.threadID))
        train(self.mode, self.threadID)
        print("退出线程：state {0}".format(self.threadID))



if __name__ == "__main__":
    """
    threadLFA = TrainThread("Mode -- {0} Thread".format("LFA"), "LFA")
    threadtable = TrainThread("Mode --  {0} Thread".format("table"), "table")

    threadLFA.start()
    threadtable.start()

    threadLFA.join()
    threadtable.join()

    print("Train is done...")
    """

    train("oracle")