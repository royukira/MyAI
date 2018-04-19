"""
A toy model for the experiment of analysis of memory effects

Action: a = {+1, -1}
State: s' = s + a
Reward: r = beta1 * (s + a) + bate2
Optimal Q-value: Q(s,a; beta) = r = beta1 * (s + a) + beta2;    # beta2 is more like a bias
Expected Q-value: Q_hat(s,a; theta) = theta1 * (s + a) + theta2;
TD-error:  e = [r(s,a; beta) + gamma * maxQ(s',a'; theta) - Q(s,a theta)]

"""
import numpy as np

class Q_learning_ER:

    def __init__(self, learnRate, gamma, batchSize):
        self.alpha = learnRate  # learning rate
        self.gamma = gamma  # Discount factor
        self.batchSize = batchSize

        # Initialize parameters randomly
        self.theta1 = np.random.uniform()
        self.theta2 = np.random.uniform()


    def maxQvaule(self,s):
        pass



