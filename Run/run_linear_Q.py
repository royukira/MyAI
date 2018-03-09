from Q_learning.Q_Brain_Simply import linear_Q

if __name__ == "__main__":

    """Setting"""
    N_STATES = 5  # the length of the 1 dimensional world
    ACTIONS = [1, 2]  # available actions 1:right, 2:wrong
    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 1-1/N_STATES  # discount factor
    MAX_EPISODES = 100  # maximum episodes
    FRESH_TIME = 0.3  # fresh time for one move



    train_model = linear_Q(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA, discountFactor=GAMMA,
                           Max_episode=MAX_EPISODES)
    print(train_model.linear_train())