"""
My Algorithm: apER -- Adaptive (memory) - Prioritized Experience Replay
"""

import numpy as np
from Q_learning.Q_Brain_Simply import linear_Q
from Memory import TimeTag as tt
from Memory import oraSumTree as ost
from matplotlib import pyplot as plt

np.random.seed(1)


class apER_Memory(object):

    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.capacity = 0
        self.tree = ost.easySumTree()
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  # clipped abs error
        self.data_pointer = 0

        # Memory System: Two main part; independent;
        #
        # sumTree Structure
        # Tree part:
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        # ID frame part:
        # [-------------- id frame-------------]
        #             size: capacity
        # Actually data is stored in DB and use a parameter 'max_capacity' to simulate the buffer

        self.timetag = tt.timeTag(max_capacity)
        # the input parameter is the initial size of the time table;
        # automatically resize (enlarge) when the capacity is not enough

        # timeTag Structure
        #
        # Note: Time node Ti = [ previous | Ti | Transition Set in ith time step | next time node]
        #
        # [ ---- Hash map ---- ]  [ -------------------- Sub-map items -------------------- ]
        # [         0          ]  [                   [ T0 | Time node T0 ]                 ]
        # [         1          ]  [                   [ T1 | Time node T1 ]                 ]
        # [         2          ]  [                   [ T2 | Time node T2 ]                 ]
        # [         3          ]  [                   [ T3 | Time node T3 ]                 ]
        # [        ...         ]  [                           ...                           ]
        # [         t          ]  [                   [ Tt | Time node Tt ]                 ]

    def store(self, transition, time_step):
        # oraSumTree Part
        max_p = self.tree.max_priority()
        if max_p is None:
            max_p = self.abs_err_upper

        if self.capacity < self.max_capacity:
            self.tree.add([max_p], [transition])  # set the max p for new p
            self.data_pointer += 1
            self.capacity += 1
        else:
            # overlap the old transitions
            self.tree.db.cover(self.tree.tableName, self.data_pointer, transition, max_p)
            self.data_pointer += 1

        if self.data_pointer >= self.max_capacity:
            self.data_pointer = self.tree.db.min_idx(self.tree.tableName)

        # Time Tag Part
        if self.timetag.get_node(time_step) is False:
            # Do not exist the node of the current learning time step
            self.timetag.add_node(time_step, [transition])
        else:
            # Exist the node of the current learning time step
            self.timetag.insert_transition(transition)

    def sample(self, n):
        # First, construct the sum tree
        self.tree.construct_tree()

        # Initial batch index, batch memory, IS weights
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 4)), np.empty((n, 1))
        pri_seg = self.tree.total_p() / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p()
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data

        # clean the tree
        self.tree.clean_tree()

        return b_idx, b_memory, ISWeights

    def sample_k_oldest(self, k):
        """
        Find k oldest transitions for calculating the old target error before the adjustment of memory

        The sampled transitions will be pop out from the original tim e node

        :param k: the number of samples
        :return: a list of oldest transitions
        """
        oldest_transitions = self.timetag.select_k_oldest(k)
        return oldest_transitions

    def batch_update(self, id, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(id, ps):
            self.tree.update(ti, p)

    def enlarge(self, k):
        self.max_capacity += k
        self.data_pointer = self.capacity  # reset the data pointer back

    def ost_shrink(self, k, removal_id):
        # Delete k oldest transitions from oraSumTree
        for id in removal_id:
            self.tree.remove(id)
        self.max_capacity -= k
        self.capacity -= k
        self.data_pointer -= k

    def tt_shrink(self, tt_transitions):
        for t in tt_transitions:
            for time, idx in t:
                self.timetag.remove_transition(time,idx)
                # if all transition in the time node have been deleted (i.e. all are None)
                # it will automatically delete the time node, but the sub-map is reserved

    def tt_insert(self, tt_transtions):
        # after using the sampled transitions, the transitions must be inserted to the latest time node
        # except those transitions that need to be deleted
        for t in tt_transtions:
            self.timetag.insert_transition(t)


class aER_Memory:

    def __init__(self,memorysize):
        self.ms =memorysize
        self.memory = np.zeros((self.ms, 4))
        self.memory_counter = 0

    def memorySize(self):
        return self.ms

    def store(self,e):
        """
        Store the experience into the memory D
        :param e: e = (s,a,r,s_)
        :return: None
        """
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        index = self.memory_counter % self.ms
        self.memory[index,:] = e

        self.memory_counter += 1


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
