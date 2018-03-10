from Q_learning.Q_Brain_Simply import linear_Q
from Q_learning.Q_Brain_Simply import QBrainSimply
import numpy as np

if __name__ == "__main__":

    """Setting"""
    N_STATES = 25  # the length of the 1 dimensional world
    ACTIONS = [1, 2]  # available actions 1:right, 2:wrong
    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 1-(1/N_STATES-1)  # discount factor
    MAX_EPISODES = 100  # maximum episodes

    """
    Load the memory
    """
    memory = np.load("./blind_cliffwalk_experience/memory_4.npy")



    train_model = linear_Q(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA, discountFactor=GAMMA,
                           Max_episode=MAX_EPISODES)

    train_w = train_model.batch_linear_train(memory=memory, batchSize=8, max_epoch=5000)

    train_model.test_policy(train_w, 15)


    train = QBrainSimply(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA, discountFactor=GAMMA,
                           Max_episode=MAX_EPISODES)

    train_table = train.batch_table_train(memory,8,5000)

    print(train_table)

    train.test_policy(train_table,15)