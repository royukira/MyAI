"""
My AI experience: Maze Game

This script is about training a machine through Deep Q-learning

The Neural Network is included standard NN, CNN

"""

import numpy as np
import pandas as pd
import tensorflow as tf


class DQN_st:
    def __init__(self,
                 n_action,
                 n_features,
                 learnRate=0.01,
                 discountFactor=0.9,
                 greedy_max=0.95,
                 memory_size=500,
                 no_op_max = 200,
                 learnIter=5,
                 TN_learn_step = 10,
                 minibatch_size = 32,
                 episode_max = 1000,
                 greedy_increment = True):
        """
        Initialize the DQN Brain of the Agent
          - initialize the hyperparameters
          - initialize the memory set
          - build the neural network

        :param n_action: number of actions
        :param n_features:
        :param learnRate: learning rate alpha
        :param discountFactor: discount factor gamma
        :param greedy_max: the maximum of the greedy value
        :param memory_size: the maximum size of the memory
        :param no_op_max: the maximum step of doing no-meaning-action at the start of the episode
        :param learnIter: every x step update the neural network (mainly update Q network)
        :param TN_learn_step: every y step update the target network
        :param minibatch_size: for sgd
        :param episode_max:the maximum iteration of episode
        :param greedy_increment: is the greedy incremented according to the learning step
        """
        self.n_action = n_action
        self.n_features = n_features
        self.learnRate = learnRate
        self.discountFactor = discountFactor
        self.greedy = greedy_max if greedy_increment is False else self.greedy_increment() #TODO
        self.memory_size = memory_size
        self.no_op_max = no_op_max
        self.learnIter = learnIter
        self.TN_learn_iter = TN_learn_step
        self.minibatch_size = minibatch_size
        self.episode_max = episode_max

        # Counter
        self.total_learning_step = 0
        self.memory_counter = 0

        # Initialization of memory
        # memory is a matrix [memory_size, exp_size]
        # exp_size = n_features*2+2
        # a experience includes 2 states with n_features, and a, r
        self.memory = np.zeros((self.memory_size, n_features*2+2))

        # Build the neural network
        n_hidden_units = [4,4,4]
        self._build_NN(n_hidden_units) # TODO
        t_para = tf.get_collection('target_net_params')
        q_para = tf.get_collection("q_net_params")
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_para, q_para)]  # for t,e 把t_para赋值于t, e:= q_para

        # Open the session
        self.sess = tf.Session()

        # Initialize the hyper-parameters of the NN
        self.sess.run(tf.global_variables_initializer())

        # Log
        tf.summary.FileWriter("logs/", self.sess.graph)

        # Cost
        self.cost_his = []

    def greedy_increment(self, incrementRate=0.001, greedy_max=0.95):
        """
        self-increment greeedy : epsilon += incrementRate / (epsilon + c)^2

        :param incrementRate: global increment rate  /  default: 0.001
        :param greedy_max: the max greedy  /  default: 0.95
        :return: None
        """
        if self.total_learning_step == 0:
            self.greedy = 0.1
        elif self.greedy < greedy_max:
            self.greedy += incrementRate/np.square(self.greedy+0.00001)  # 0.00001 ensures the denominator will not be 0
        else:
            self.greedy = greedy_max

    def _build_NN(self,n_hidden_units):
        # ------------- Q-network ---------------
        lays_info = {}
        with tf.name_scope('Input_of_Q_net'):
            self.s = tf.placeholder(dtype=tf.float32,shape=[None, self.n_features], name="s")
        with tf.name_scope("Target_value"):
            self.target_value = tf.placeholder(tf.float32, [None, self.n_action], name='Q_target')  # for calculating loss
        with tf.name_scope("para_of_Q_net"):
            # The first layer
            w1 = tf.truncated_normal([2, n_hidden_units[0]], stddev=0.1)
            b1 = tf.Variable(tf.zeros([n_hidden_units[0]]))
            # collections is used later when assign to target net
            tf.add_to_collection("q_net_params", w1)  # add w1 to the collection "q_net_para"
            tf.add_to_collection("q_net_params", b1)  # add b1 to the collection "q_net_para"
            if (1 in lays_info) is False:
                lays_info.setdefault(1, ("Layer1", w1, b1, tf.nn.relu))

            # The 2nd layer
            w2 = tf.truncated_normal([n_hidden_units[0], n_hidden_units[1]], stddev=0.1)
            b2 = tf.Variable(tf.zeros([n_hidden_units[1]]))
            # collections is used later when assign to target net
            tf.add_to_collection("q_net_params", w2)  # add w2 to the collection "q_net_para"
            tf.add_to_collection("q_net_params", b2)  # add b2 to the collection "q_net_para"
            if (2 in lays_info) is False:
                lays_info.setdefault(2, ("Layer2", w2, b2, tf.nn.relu))

            # The output layer
            w3 = tf.truncated_normal([n_hidden_units[1], self.n_action], stddev=0.1)
            b3 = tf.Variable(tf.zeros([self.n_action]))
            # collections is used later when assign to target net
            tf.add_to_collection("q_net_params", w3)  # add w3 to the collection "q_net_para"
            tf.add_to_collection("q_net_params", b3)  # add b3 to the collection "q_net_para"
            if (3 in lays_info) is False:
                lays_info.setdefault(3, ("OutputLayer", w3, b3, tf.nn.relu))

        # Create the Q Neural Network
        from FastBuildNet.FBN import createLayer
        self.predict_value = createLayer(self.s, lays_info)

        # ------------- Target-network ---------------
        target_lays_info = {}
        with tf.name_scope("Input_of_target_net"):
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name="s_")

        with tf.name_scope("para_of_target_net"):
            # The first layer
            w_1 = tf.truncated_normal([2, n_hidden_units[0]], stddev=0.1)
            b_1 = tf.Variable(tf.zeros([n_hidden_units[0]]))
            # collections is used later when assign to target net
            tf.add_to_collection("target_net_params", w_1)  # add w1 to the collection "q_net_para"
            tf.add_to_collection("target_net_params", b_1)  # add b1 to the collection "q_net_para"
            if (1 in target_lays_info) is False:
                target_lays_info.setdefault(1, ("Target_Layer1", w_1, b_1, tf.nn.relu))

            # The 2nd layer
            w_2 = tf.truncated_normal([n_hidden_units[0], n_hidden_units[1]], stddev=0.1)
            b_2 = tf.Variable(tf.zeros([n_hidden_units[1]]))
            # collections is used later when assign to target net
            tf.add_to_collection("target_net_params", w_2)  # add w2 to the collection "q_net_para"
            tf.add_to_collection("target_net_params", b_2)  # add b2 to the collection "q_net_para"
            if (2 in target_lays_info) is False:
                target_lays_info.setdefault(2, ("Target_Layer2", w_2, b_2, tf.nn.relu))

            # The output layer
            w_3 = tf.truncated_normal([n_hidden_units[1], self.n_action], stddev=0.1)
            b_3 = tf.Variable(tf.zeros([self.n_action]))
            # collections is used later when assign to target net
            tf.add_to_collection("target_net_params", w_3)  # add w3 to the collection "q_net_para"
            tf.add_to_collection("target_net_params", b_3)  # add b3 to the collection "q_net_para"
            if (3 in target_lays_info) is False:
                target_lays_info.setdefault(3, ("Target_OutputLayer", w_3, b_3, tf.nn.relu))

        # Create the Target Neural Network
        self.q_next = createLayer(self.s_, target_lays_info)

        # -------- Loss Function ------
        with tf.name_scope("The_mean-square_Loss_Function"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_value,self.predict_value))
            tf.summary.scalar('The_mean-square_Loss_Function', self.loss)

        # -------- Train (optimizer) ---------
        with tf.name_scope("Training"):
            self.optimizer = tf.train.RMSPropOptimizer(self.learnRate).minimize(self.loss)  # RMSprop
            # RMSprop 是自适应学习率优化算法
            # 其实RMSprop依然依赖于全局学习率
            # 但对学习率有个约束作用
            # 详细：https://zhuanlan.zhihu.com/p/22252270
            #      http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
            #      http://ruder.io/optimizing-gradient-descent/index.html#rmsprop

    def store_exp(self,e):
        """
        Store the experience into the memory D
        :param e: e = (s,a,r,s_)
        :return: None
        """
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = e

        self.memory_counter += 1

    def choose_action(self,observation):
        gv = np.random.uniform()

        # preprocess the observation
        # observation == [x,y], shape == (2,)
        # observation[np.newaxis,:] == [[x y]], shape == (1,2)
        observation = observation[np.newaxis, :]

        # forward feed the observation and get q value for every actions
        if gv <= self.greedy:
            # use NN to cal the action_value
            action_values = self.sess.run(self.predict_value,feed_dict={self.s:observation})
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self.n_action)

        return action

    def learn(self):
        """
        Train the Neural Network
        :return:
        """

        # check if update the parameter of target network with the old parameter of Q network
        if self.total_learning_step % self.learnIter == 0:
            self.sess.run(self.replace_target_op)
            print("--> The parameter of target network is updated")

        # pull out a batch of sample(i.e. experience) from memory set D
        # Need to improve!!!! # TODO # To Research
        if self.memory_counter < self.memory_size:
            sample_index = np.random.choice(self.memory_counter,size=self.minibatch_size)
        else:
            sample_index = np.random.choice(self.memory_size,size=self.minibatch_size)
        batch_sample = self.memory[sample_index,:]

        q_next, predict_value = self.sess.run(
            [self.q_next, self.predict_value],
            feed_dict={
                self.s_: batch_sample[:, -self.n_features:],  # fixed params
                self.s: batch_sample[:, :self.n_features],  # newest params
            })

        # copy the matrix of predict_value to target_value.
        # bring the matrix of target value into correspondence with the one of predict_value
        # so that change the target value w.r.t predict_value
        target_value = predict_value.copy()

        # retrieve action (a) and the reward (r) of state (s) from the experience (samples)
        batch_sample_index =  np.arange(self.minibatch_size,dtype=np.int32)
        eval_act = batch_sample[:, self.n_features].astype(int)  # eval_act = [a0 a1 a2 ...]
        reward = batch_sample[:, self.n_features+1]  # reward = [r0 r1 r2 ... ]

        # check if there are any terminates(terminates or hells)
        terminates_index = np.where(batch_sample[:, -self.n_features:] == "Terminate")[0]
        ter_act = eval_act[terminates_index]  # the corresponding action with terminate
        ter_reward = reward[terminates_index]  # the corresponding reward with terminate
        # delete the terminates' indices from all samples' indices
        batch_sample_index = np.delete(batch_sample_index, terminates_index)
        eval_act = np.delete(eval_act, terminates_index)
        reward = np.delete(reward, terminates_index)

        # Calculate Q-Target Value
        # terminate
        target_value[terminates_index, ter_act] = ter_reward
        # NOT terminate
        target_value[batch_sample_index,eval_act] = reward + self.greedy * np.max(q_next, axis=1)

        # Loss function and optimization -- begin to study (update the weights)
        opt, self.L = self.sess.run([self.optimizer, self.loss],
                               feed_dict={
                                   self.s: batch_sample[:, :self.n_features],
                                   self.target_value: target_value
                               })
        self.cost_his.append(self.L)

        # increasing greedy 如果放在探索step里增加效果？？
        self.greedy_increment()
        self.total_learning_step += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


if __name__ == '__main__':
    test = DQN_st(4,2)








