"""
My Algorithm: apER -- Adaptive (memory) - Prioritized Experience Replay
"""

import numpy as np
from Q_learning.Q_Brain_Simply import linear_Q
from matplotlib import pyplot as plt

np.random.seed(1)


class SumTree(object):
    # TODO: 需要修改 能动态改变大小，能否实现？
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Store the priorities of the transitions
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


class BpTree(object):pass # TODO: for storing T0


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

# ========= Sum Tree operations ============

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

# ========= B+ Tree operations ===============

    def get_K_oldest(self, *args):
        pass
    # TODO: select K oldest transitions using BpTree which stores the initial time of transitions

# ========= Enlarge =========

    def enlarge(self):pass
    # TODO: Enlarge the memory and trees

# ========= Shrink ==========

    def shrink(self):pass
    # TODO: Shrink the memory and trees


class apER_train(linear_Q):
    def __init__(self, numState, actions, learnRate,
                 discount, greedy, init_memorySize, batchSize,
                 k_old, forget_threshold):

        super(apER_train, self).__init__(numState, actions, greedy, learnRate, discount)

        # General parameters
        self.numState = numState - 1  # exclude terminate
        self.alpha = learnRate
        self.gamma = discount
        self.epsilon = greedy
        self.ActionSet = actions

        # Memory
        self.memorySize = init_memorySize  # the initial memory size
        self.memory = Memory(self.memorySize)  # initialize the memory buffer with initial memory size
        self.batchSize = batchSize  # the size of mini batch

        # Time Step
        self.steps = 0

        # Adaptive Part
        self.select_k = k_old
        self.forget_threshold = forget_threshold
        self.tmp_oldest = []  # for store the k oldest transitions temporarily
        self.old_td_error = 0  # the last sum of TD errors of the k oldest transitions

        # Review Part

        # Optimization
        self.gradient = 0

    def prioritized_Learn(self):pass  # TODO: this part is same as pER

    def adaptive_Memroy(self):pass  # TODO: the part of adjust the memory size and delete some old transitions

    def review(self):pass  # TODO: the part of review; rise the priority of the low prioritized transitions
