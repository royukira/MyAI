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

    def __init__(self, learnRate, gamma, numState, batchSize):
        self.alpha = learnRate  # learning rate
        self.gamma = gamma  # Discount factor
        self.batchSize = batchSize
        self.numState = numState
        # Initialize parameters randomly
        #self.weights = np.random.uniform(size=(2, self.numState))  # the shape of theta is (2, #states)
        self.weights = np.random.uniform(size=(2,1))

        # Action Set
        self.actions = [-1, +1]

    def maxQvalue(self,s):
        q1 = self.weights[0] * (s + self.actions[0]) + self.weights[1]
        q2 = self.weights[0] * (s + self.actions[1]) + self.weights[1]

        if q1 > q2:
            return self.actions[0]
        else:
            return self.actions[1]

    def ERtrain(self, batchSize, memory, target):
        """Training part"""
        batchIndex = np.random.choice(memory.shape[0], size=batchSize)
        batchSample = memory[batchIndex, :]
        #gradient = np.zeros((2, self.numState))
        #distance = np.zeros((2, self.numState))
        gradient = np.zeros((2,1))

        for sample in batchSample:
            s = int(sample[0])
            a = int(sample[1])
            r = int(sample[2])
            s_ = int(sample[3])

            s_index = s-1

            if self.gamma == 0:
                # vectorization
                S_ = np.ones((2,1))
                S_[0] = s_

                # the weights of current state s
                #weight = self.weights[:, s_index][:, np.newaxis]  # s starts from 1

                # Distance (gamma=0) [beta1 - theta1; beta2-theta2]
                dist = target - self.weights  # gradient of weights
                #distance[:,  s_index] = distance[:, s_index] + dist.ravel()

                # Gradient of ith sample
                td_error = np.dot(np.transpose(S_), dist)
                gd = td_error * S_
                gradient = gradient + gd

            else:
                pass # TODO
                """
                
                # vectorization
                S_2d = np.ones((2,1))
                S_2d[0] = s_
                S_3d = np.ones((3,1))
                S_3d[0] = s_

                # weights
                weight = self.weights[:, s]

                # Distance
                dist = np.zeros((3,1))
                dist[0] = target[0] - (1 - self.gamma) * weight[0]
                dist[1] = target[1] - (1 - self.gamma) * weight[1]
                dist[2] = self.gamma * weight[0]

                # only need dist[0] and dist[1]
                distance[:, s] = distance[:, s] + dist[0:2]

                # Gradient of ith sample
                gradient[:, s] = gradient[:, s] + np.dot(np.transpose(S_3d), dist) * S_2d
                """
        increment = self.alpha * (gradient / batchSize)
        self.weights += increment
        #dist1_norm = np.linalg.norm(distance[0, :]) / batchSize * 2
        #dist2_norm = np.linalg.norm(distance[1, :]) / batchSize * 2
        grad_norm = np.linalg.norm(gradient) / batchSize
        return grad_norm














