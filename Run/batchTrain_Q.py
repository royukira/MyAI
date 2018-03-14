from Q_learning.Q_Brain_Simply import linear_Q
from Q_learning.Q_Brain_Simply import QBrainSimply
from Q_learning.Q_Brain_Simply import oracle_Q
from Q_learning.Q_Brain_Simply import Memory as my
from Q_learning.QForget import priority_train
import numpy as np
import threading
from training_env import Simply_Teasure_Game as stg
from matplotlib import pyplot as plt


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

    if mode == "priority":
        tableName = "PRIORITY{0}".format(N_STATES-1)

        train = priority_train(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                             discountFactor=GAMMA,
                             Max_episode=MAX_EPISODES,
                               memorySize=memory.shape[0])

        train.load_memory(memory)
        train_w, step = train.priorityTrain(batchSize, max_epoch)

        train.plot(step)

        print("--> Thread: {0} well trained...\n--> {0} Start to test...".format(threadID))
        print(train_w)

        train.test_policy(train_w, 15)



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

def train_with_play(batchSize,numState,RL,memory=None,mode=None):
    total_steps = 0
    steps = []
    episodes = []
    memory = my(30)
    for i_episode in range(20):
        s=0
        stg.update_env(s, i_episode, total_steps, numState)
        while True:
            # env.render()
            random_choose = np.random.uniform()

            q,a = RL.easy_find_max_q(s)


            S_Next, R = stg.get_env_feedback(s,a,numState)

            transition = np.hstack((s, a, R, S_Next))
            if mode == "l":
                memory.store_exp(transition)
            if mode == "p":
                RL.memory.store(transition)

            if total_steps >batchSize:
                print("Learning...")
                if mode == "p":
                    RL.priorityTrain(batchSize)
                if mode == "l":
                    RL.batch_linear_train(memory.memory,batchSize)


            if R == 1:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            if S_Next == -1:
                s = 0
            else:
                s = S_Next
            #stg.update_env(s, i_episode, total_steps, numState)
            total_steps += 1

    return np.vstack((episodes, steps))




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

    #train("priority")

    """Setting"""
    N_STATES = 500 # the length of the 1 dimensional world
    ACTIONS = [1, 2]  # available actions 1:right, 2:wrong
    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 1 - (1 / N_STATES - 1)  # discount factor
    MAX_EPISODES = 100  # maximum episodes
    batchSize = 30
    max_epoch = 600

    """
    Load the memory
    """
    memory = np.load("/Users/roy/Documents/GitHub/MyAI/Run/blind_cliffwalk_experience/memory_25.npy")

    test = priority_train(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                          discountFactor=GAMMA,
                          Max_episode=MAX_EPISODES,
                          memorySize=memory.shape[0])

    test2 = train_model = linear_Q(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                               discountFactor=GAMMA,
                               Max_episode=MAX_EPISODES)
    #test.load_memory(memory)

    step = train_with_play(batchSize,N_STATES,test,mode="p")
    step2 = train_with_play(batchSize,N_STATES,test2,memory=memory,mode="l")


    plt.figure(1)
    plt.plot(step2[0, :], step2[1, :] - step2[1, 0], c='b', label='linear FA')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()

    plt.figure(2)
    plt.plot(step[0, :], step[1, :] - step[1, 0], c='r', label='prioritized memory')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()