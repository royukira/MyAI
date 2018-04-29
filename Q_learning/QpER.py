"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)
Refer to Morvan Zhou's tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
from Q_learning.Q_Brain_Simply import linear_Q
from matplotlib import pyplot as plt

np.random.seed(1)



class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class priority_train(linear_Q):

    def __init__(self, numState, ActionSet, greedy, learnRate, discountFactor, memorySize, Max_episode=None):
        super(priority_train, self).__init__(numState, ActionSet, greedy, learnRate, discountFactor, Max_episode)
        """initial Value"""
        self.numState = numState - 1  # 减去terminal
        self.ActionSet = ActionSet
        self.greedy = greedy
        self.learnRate = learnRate
        self.discountFactor = discountFactor
        self.memorySize = memorySize
        self.memory = Memory(self.memorySize)
        self.gradient = 0

    def load_memory(self, InputMemory):
        for m in InputMemory:
            self.memory.store(m)

    def priorityTrain(self,batchSize):
        """Training part"""

        from Utility_tool.laplotter import LossAccPlotter
        """
        Repeat the episode until s gets to the rightmost position (i.e. get the treasure)
        """
        is_terminal = False  # ending signal
        save_path = "/Users/roy/Documents/GitHub/MyAI/Log/BCW_loss/{0}_state_priority.png".format(self.numState)
        """
        plotter = LossAccPlotter(title="Loss of Prioritized Replay with {0} states".format(self.numState),
                                 save_to_filepath=None,
                                 show_acc_plot=False,
                                 show_plot_window=False,
                                 show_regressions=False,
                                 LearnType="LFA"
                                 )
        """

        tree_idx, batch_memory, ISWeights = self.memory.sample(batchSize)
        w_increment = np.zeros((self.numState * len(self.ActionSet) + 1, 1))
        abs_error_ = np.zeros((batchSize, 1))
        batchIndex = 0

        for sample in batch_memory:
            s = int(sample[0])
            a = int(sample[1])
            r = int(sample[2])
            s_ = int(sample[3])

            if s == 0:
                if a == 1:
                    x = self.X_S_A[:, s][:, np.newaxis]
                if a == 2:
                    x = self.X_S_A[:, s + 1][:, np.newaxis]
            else:
                if a == 1:
                    x = self.X_S_A[:, s * len(self.ActionSet)][:,
                        np.newaxis]  # the x's shape is (#states * #action+1, 1)
                elif a == 2:
                    x = self.X_S_A[:, s * len(self.ActionSet) + 1][:,
                        np.newaxis]  # the x's shape is (#states * #action+1, 1)

            q_predict = np.dot(np.transpose(x), self.W)

            """
            Calculate the target
            """
            if s_ == -1:
                self.target_error = r - q_predict
            else:
                max_Q, max_A = self.easy_find_max_q(s_)
                self.target_error = r + self.discountFactor * max_Q - q_predict

            abs_error = np.absolute(self.target_error)

            w_increment += ISWeights[batchIndex] * self.target_error * x

            abs_error_[batchIndex] = abs_error

            batchIndex += 1

        """
        update 
        """
        self.W += self.learnRate * w_increment

        self.gradient = np.linalg.norm(w_increment) / batchSize  # gradient of one learning step
        self.memory.batch_update(tree_idx, abs_error_)

        print("--> Linear Q-learning with Prioritized ER's gradient: {0}\n".format(self.gradient))
        print("==============================================\n")

        return self.W, self.gradient

    def plot(self,step):
        plt.plot(step[0, :], step[1, :] - step[1, 0], c='b', label='prioritized replay')
        print(step[1, :])
        print(step[1, 0])
        plt.legend(loc='best')
        plt.ylabel('total training step')
        plt.xlabel('episode')
        plt.grid()
        plt.show()



if __name__ == '__main__':
    """Setting"""
    N_STATES = 4  # the length of the 1 dimensional world
    ACTIONS = [1, 2]  # available actions 1:right, 2:wrong
    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 1 - (1 / N_STATES - 1)  # discount factor
    MAX_EPISODES = 100  # maximum episodes
    batchSize = 3
    max_epoch = 600

    """
    Load the memory
    """
    memory = np.load("/Users/roy/Documents/GitHub/MyAI/Run/blind_cliffwalk_experience/memory_4.npy")

    test = priority_train(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                          discountFactor=GAMMA,
                          Max_episode=MAX_EPISODES,
                          memorySize = memory.shape[0])

    test.load_memory(memory)
    print(test.memory.tree.data)
