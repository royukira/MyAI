from Q_learning.Q_Brain_Simply import QBrainSimply

if __name__ == "__main__":

    """Setting"""
    N_STATES = 7  # the length of the 1 dimensional world
    ACTIONS = ['left', 'right']  # available actions
    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 0.9  # discount factor
    MAX_EPISODES = 13  # maximum episodes
    FRESH_TIME = 0.3  # fresh time for one move

    """Create the Q Brain"""
    Q_brain = QBrainSimply(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA, discountFactor=GAMMA,
                           Max_episode=MAX_EPISODES)
    print(Q_brain.train_brain())