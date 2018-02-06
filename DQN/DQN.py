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
                 greedy_max=0.9,
                 memory_size=500,
                 no_op_max = 200,
                 learnIter=5,
                 TN_learn_step = 50,
                 minibatch_size = 32,
                 episode_max = 1000,
                 greedy_increment = False):
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

    def para_summaries(self, para, name):
        """
        Attach a lot of summaries to a Tensor.
        --------------------------------------
        :param para: Wi or Bi (ith layer)
        :param name: Layer name + Parameter name
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(para)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(para - mean)))
            tf.summary.scalar('sttdev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(para))
            tf.summary.scalar('min/' + name, tf.reduce_min(para))
            tf.summary.histogram(name, para)

    def _build_NN(self,n_hidden_units):
        # ------------- Q-network ---------------
        lays_info = {}
        with tf.name_scope('Input_of_Q_net'):
            self.s = tf.placeholder(dtype=tf.float32,shape=[None, self.n_features], name="s")

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
        self.target_value = createLayer(self.s_, target_lays_info)

        # -------- Loss Function ------
        with tf.name_scope("The_mean-square_Loss_Function"):
            loss = tf.reduce_mean(tf.squared_difference(self.predict_value,self.target_value))
            tf.summary.scalar('The_mean-square_Loss_Function', loss)

        # -------- Train (optimizer) ---------
        with tf.name_scope("Training"):
            self.optimizer = tf.train.RMSPropOptimizer(self.learnRate).minimize(loss)  # RMSprop

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






if __name__ == '__main__':
    test = DQN_st(4,2)








