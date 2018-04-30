"""
My Algorithm: apER -- Adaptive (memory) - Prioritized Experience Replay
"""

import numpy as np
import random
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
        # [-------------- data frame-------------]
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
        """
        Firstly, store the transitions to time table and get back a set of time-marked transitions
        :param transition: a transition at time_step
        :param time_step: the learning time step
        :return:
        """
        # Time Tag Part
        if self.timetag.get_node(time_step) is False:
            # Do not exist the node of the current learning time step
            marked_t = self.timetag.add_node(time_step, [transition])  # input a list; return a list
            marked_t = marked_t[0]  # convert back to a transition from a list
        else:
            # Exist the node of the current learning time step
            marked_t = self.timetag.insert_transition(transition)  # input a transition (array); output a transition

        # oraSumTree Part
        max_p = self.tree.max_priority()
        if max_p is None:
            max_p = self.abs_err_upper

        if self.capacity < self.max_capacity:
            self.tree.add([max_p], [marked_t])  # set the max p for new p
            self.data_pointer += 1
            self.capacity += 1
        else:
            # overlap the old transitions
            self.tree.db.cover(self.tree.tableName, self.data_pointer, marked_t, max_p)
            self.data_pointer += 1

        if self.data_pointer >= self.max_capacity:
            self.data_pointer = self.tree.db.min_idx(self.tree.tableName)

    def sample(self, n):
        # First, construct the sum tree
        self.tree.construct_tree()

        # Initial batch index, batch memory, IS weights
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 6)), np.empty((n, 1))
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

        The sampled transitions do not need to be pop out from the original time node

        Since there are no any update for transition at the adjustment stage

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
            if p[0] == 0:
                print('test')
            self.tree.update(ti, p[0])

    def tid_update(self, id, new_tid):
        time = new_tid[4]
        idx = new_tid[5]
        self.tree.update_tid(id, time, idx)

    def enlarge(self, k):
        self.max_capacity += k
        self.data_pointer = self.capacity  # reset the data pointer back

    def ost_shrink(self, k, transitions):
        # Delete k oldest transitions from oraSumTree
        for i in transitions:
            time = i[4]
            idx = i[5]
            self.tree.remove_tid(time, idx)

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

    # TODO: under construction

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
                 k_old, forget_threshold=None):

        super(apER_train, self).__init__(numState, actions, greedy, learnRate, discount)

        # General parameters
        self.numState = numState - 1  # exclude terminate
        self.alpha = learnRate  # learning rate
        self.gamma = discount  # discount factor
        self.epsilon = greedy  # epsilon-greedy policy
        self.ActionSet = actions  # the action set

        # Memory
        self.memorySize = init_memorySize  # the initial memory size
        self.memory = apER_Memory(self.memorySize)  # initialize the memory buffer with initial memory size
        self.batchSize = batchSize  # the size of mini batch

        # Learning time Step
        self.learning_steps = 0

        # Adaptive Part
        self.select_k = k_old
        self.forget_threshold = forget_threshold
        self.tmp_oldest = []  # for store the k oldest transitions temporarily
        self.last_old_td_error = 0  # the last sum of TD errors of the k oldest transitions
        self.cur_old_td_error = 0

        # Review Part

        # Optimization
        self.gradient = 0

    def prioritized_Learn(self):  # TODO: this part is same as pER
        """
        :return:
        """
        batch_idx, batch_memory, ISWeights = self.memory.sample(self.batchSize)
        w_increment = np.zeros((self.numState * len(self.ActionSet) + 1, 1))
        abs_error_ = np.zeros((self.batchSize, 1))
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

            if abs_error == 0:
                print()

            w_increment += ISWeights[batchIndex] * self.target_error * x

            abs_error_[batchIndex] = abs_error

            batchIndex += 1

        """
        update 
        """
        self.W += self.learnRate * w_increment

        self.gradient = np.linalg.norm(w_increment) / self.batchSize  # gradient of one learning step

        # Update the priorities stored in the DB
        self.memory.batch_update(batch_idx, abs_error_)

        print("--> Linear Q-learning with Prioritized ER's gradient: {0}\n".format(self.gradient))
        print("==============================================\n")

        # time
        self.learning_steps += 1

        for id, sample in zip(batch_idx, batch_memory):
            T = int(sample[4])
            idx = int(sample[5])

            # =================== For updating Time Tag =====================
            # Update the transitions to the latest time node
            # Firstly, pop out the sampled transitions from old time step and put them to the latest time node
            self.memory.timetag.remove_transition(T, idx)
            # Secondly, insert them into the newest time node
            if self.memory.timetag.get_node(self.learning_steps) is False:
                # Do not exist the node of the current learning time step
                #test=sample[0:4]
                new_tid = self.memory.timetag.add_node(self.learning_steps, [sample[0:4]])  # input a list; return a list
                new_tid = new_tid[0]
            else:
                # Exist the node of the current learning time step
                new_tid = self.memory.timetag.insert_transition(sample[0:4])
            # ===============================================================
            self.memory.tid_update(id, new_tid)

        return self.W, self.gradient

    def adaptive_Memroy(self, k):  # TODO: the part of adjust the memory size and delete some old transitions
        """
        Adjust the memory size
        :param k: increment k or decrement k
        :return:
        """
        # Step 1: sample n oldest transitions;
        # in the same time step, sample transitions randomly
        self.tmp_oldest = self.memory.timetag.select_k_oldest(self.select_k)

        # Step 2: Calculate the sum of TD-error of n-oldest transitions
        self.calc_sum_error()

        # Step 3: Compare the last n-oldest TD-error with the current n-oldest TD-error
        if self.cur_old_td_error > self.last_old_td_error:
            self.memory.enlarge(k)
            self.last_old_td_error = self.cur_old_td_error
            print("--> learning step {0}: Enlarging the memory".format(self.learning_steps))
        else:
            # TODO: Think: how to decided the removal transitions? Randomly choose from k-oldest set? or using F-value
            removal = random.sample(self.tmp_oldest, k)
            # shrink the oraSumTree
            self.memory.ost_shrink(k, removal)
            # shrink the time tag
            self.memory.tt_shrink(removal)

            # re-sample n-oldest
            self.cur_old_td_error = 0  # reset to 0; re-calculate
            self.tmp_oldest = self.memory.timetag.select_k_oldest(self.select_k)
            # recompute the error
            self.calc_sum_error()
            self.last_old_td_error = self.cur_old_td_error
            print("--> learning step {0}: Shrinking the memory".format(self.learning_steps))

            #k_compensation = self.memory.timetag.select_k_oldest(k)

    def calc_sum_error(self):
        for t in self.tmp_oldest:
            s = int(t[0])
            a = int(t[1])
            r = int(t[2])
            s_ = int(t[3])

            if s == 0:
                if a == 1:
                    xx = self.X_S_A[:, s][:, np.newaxis]
                if a == 2:
                    xx = self.X_S_A[:, s + 1][:, np.newaxis]
            else:
                if a == 1:
                    xx = self.X_S_A[:, s * len(self.ActionSet)][:,
                         np.newaxis]  # the xx's shape is (#states * #action+1, 1)
                elif a == 2:
                    xx = self.X_S_A[:, s * len(self.ActionSet) + 1][:,
                         np.newaxis]  # the xx's shape is (#states * #action+1, 1)

            q_predict = np.dot(np.transpose(xx), self.W)

            """
            Calculate the target
            """
            if s_ == -1:
                self.target_error = r - q_predict
            else:
                max_Q, max_A = self.easy_find_max_q(s_)
                self.target_error = r + self.discountFactor * max_Q - q_predict

            self.cur_old_td_error += np.absolute(self.target_error)[0][0]

    def review(self):pass  # TODO: the part of review; rise the priority of the low prioritized transitions
