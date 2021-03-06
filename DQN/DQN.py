"""
My AI experience: Maze Game

This script is about training a machine through Deep Q-learning

The Neural Network is included standard NN

"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class DQN_st:
    def __init__(self,
                 n_action,
                 n_features,
                 learnRate=0.01,
                 discountFactor=0.9,
                 greedy_max=0.9,
                 memory_size=1000,
                 TN_learn_step = 100,
                 minibatch_size = 200,
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
        # Counter
        self.total_learning_step = 0
        self.memory_counter = 0

        # Config
        self.n_action = n_action
        self.n_features = n_features
        self.learnRate = learnRate
        self.discountFactor = discountFactor
        if greedy_increment is False:
            self.greedy = greedy_max
        else:
            self.greedy_increment()
        self.memory_size = memory_size
        self.TN_learn_iter = TN_learn_step
        self.minibatch_size = minibatch_size

        # Initialization of memory
        # memory is a matrix [memory_size, exp_size]
        # exp_size = n_features*2+2
        # a experience includes 2 states with n_features, and a, r
        self.memory = np.zeros((self.memory_size, n_features*2+2))

        # Build the neural network
        n_hidden_units = [10,10,4]
        self._build_NN(n_hidden_units)
        t_para = tf.get_collection('target_net_params')
        q_para = tf.get_collection("q_net_params")
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_para, q_para)]  # for t,e 把t_para赋值于t, e:= q_para

        # Open the session
        self.sess = tf.Session()

        # Initialize the hyper-parameters of the NN
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        # LOG
        self.loss_merged = tf.summary.merge([self.loss_scalar])
        self.Qnetwork_merged = tf.summary.merge(self.Qpara_sum_list)
        self.Tnetwork_merged = tf.summary.merge(self.TargetPara_sum_list)
        self.train_writer = tf.summary.FileWriter('/Users/roy/Documents/GitHub/MyAI/DQN_LOG', self.sess.graph)

        # Reward percentage (for plot)
        self.Rmemory_plot = []
        self.Rsample_plot = []

    def greedy_increment(self, incrementRate=0.0000003, greedy_max=0.9):
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

    def save_the_model(self,sess):
        self.saver = (sess,'/Users/roy/Documents/GitHub/MyAI/DQN_MODEL')

    def _build_NN(self,n_hidden_units):
        # import
        from FastBuildNet.FBN import createLayer
        # ------------- Q-network ---------------
        lays_info = {}
        with tf.variable_scope('Input_of_Q_net'):
            self.s = tf.placeholder(dtype=tf.float32,shape=[None, self.n_features], name="s")
        with tf.variable_scope("Target_value"):
            self.target_value = tf.placeholder(tf.float32, [None, self.n_action], name='Q_target')  # for calculating loss
        with tf.variable_scope("Q_Network"):
            c_names, w_initializer, b_initializer = \
                ['q_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            # The first layer
            with tf.variable_scope("layer1"):
                w1 = tf.get_variable('w1', [self.n_features, n_hidden_units[0]], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, n_hidden_units[0]], initializer=b_initializer, collections=c_names)
                if (1 in lays_info) is False:
                    lays_info.setdefault(1, ("Layer1", w1, b1, tf.nn.relu))

            # The 2nd layer
            with tf.variable_scope("layer2"):
                w2 = tf.get_variable('w2', [n_hidden_units[0], n_hidden_units[1]], initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable('b2', [1, n_hidden_units[1]], initializer=b_initializer, collections=c_names)
                if (2 in lays_info) is False:
                    lays_info.setdefault(2, ("Layer2", w2, b2, tf.nn.relu))

            # The output layer
            with tf.variable_scope("Output_Layer"):
                w3 = tf.get_variable('w3', [n_hidden_units[1], self.n_action], initializer=w_initializer,
                                     collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_action], initializer=b_initializer, collections=c_names)
                if (3 in lays_info) is False:
                    lays_info.setdefault(3, ("OutputLayer", w3, b3, None))

        # Create the Q Neural Network
            with tf.name_scope("Predict_value"):
                self.predict_value, self.Qpara_sum_list = createLayer(self.s, lays_info)

        """"# ------------- Target-network ---------------"""
        target_lays_info = {}
        with tf.variable_scope("Input_of_target_net"):
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name="s_")
        with tf.variable_scope("Target_Network"):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # The first layer
            with tf.variable_scope("layer1"):
                w_1 = tf.get_variable('w1', [self.n_features, n_hidden_units[0]], initializer=w_initializer,
                                      collections=c_names)
                b_1 = tf.get_variable('b1', [1, n_hidden_units[0]], initializer=b_initializer, collections=c_names)
                if (1 in target_lays_info) is False:
                    target_lays_info.setdefault(1, ("Target_Layer1", w_1, b_1, tf.nn.relu))

            # The 2nd layer
            with tf.variable_scope("layer2"):
                w_2 = tf.get_variable('w2', [n_hidden_units[0], n_hidden_units[1]], initializer=w_initializer,
                                      collections=c_names)
                b_2 = tf.get_variable('b2', [1, n_hidden_units[1]], initializer=b_initializer, collections=c_names)
                if (2 in target_lays_info) is False:
                    target_lays_info.setdefault(2, ("Target_Layer2", w_2, b_2, tf.nn.relu))

            # The output layer
            with tf.variable_scope("Output_Layer"):
                w_3 = tf.get_variable('w3', [n_hidden_units[1], self.n_action], initializer=w_initializer,
                                      collections=c_names)
                b_3 = tf.get_variable('b3', [1, self.n_action], initializer=b_initializer, collections=c_names)
                if (3 in target_lays_info) is False:
                    target_lays_info.setdefault(3, ("Target_OutputLayer", w_3, b_3, None))

        # Create the Target Neural Network
            with tf.name_scope("q_next"):
                self.q_next, self.TargetPara_sum_list = createLayer(self.s_, target_lays_info)

        # -------- Loss Function ------
        with tf.name_scope("The_Temporal_difference_error"):
            with tf.name_scope("Cost"):
                self.loss = tf.reduce_mean(tf.squared_difference(self.target_value,self.predict_value))
            self.loss_scalar = tf.summary.scalar('The_Temporal_difference_error', self.loss)

        # -------- Train (optimizer) ---------
        with tf.name_scope("Training"):
            self.optimizer = tf.train.AdadeltaOptimizer(self.learnRate).minimize(self.loss)  # RMSprop
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
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

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
        if gv < self.greedy:
            # use NN to cal the action_value
            action_values = self.sess.run(self.predict_value,feed_dict={self.s:observation})
            print(action_values)
            action = np.argmax(action_values)
            print("--> Action of {0}: {1}".format(observation,action))
        else:
            action = np.random.randint(0, self.n_action)  # 0,1,2,3 注意 action最好从0设起 例如设1，2，3，4，4是无法选到#TODO#Notice
            print("--> Randomly Action")

        return action

    def learn(self):
        """
        Train the Neural Network
        :return:
        """
        print("--> Current Learning step: {0}".format(self.total_learning_step))
        print("--> Current greedy value: {0}".format(self.greedy))

        # Calculate the number of the reward 1 in memory dataset
        rewardNum_in_memory = len(np.where(self.memory[:, self.n_features+1] == 1)[0])

        # check if update the parameter of target network with the old parameter of Q network
        if self.total_learning_step % self.TN_learn_iter == 0:
            self.sess.run(self.replace_target_op)
            print("--> The parameter of target network is updated")

        # pull out a batch of sample(i.e. experience) from memory set D
        # Need to improve!!!! # TODO # To Research
        if self.memory_counter < self.memory_size:
            Rpercent_in_memory = (rewardNum_in_memory / self.memory_counter) * 100
            print("--> reward 记忆占: {0}%".format(Rpercent_in_memory))

            # 随机抽取batch sample
            sample_index = np.random.choice(self.memory_counter,size=self.minibatch_size)
        else:
            Rpercent_in_memory = (rewardNum_in_memory / self.memory_size) * 100
            print("--> reward 记忆占: {0}%".format(Rpercent_in_memory))

            # 随机抽取batch sample
            sample_index = np.random.choice(self.memory_size,size=self.minibatch_size)

        # 记录 方便画图
        self.Rmemory_plot.append(Rpercent_in_memory)

        batch_sample = self.memory[sample_index,:]

        print("\n=================================================================")

        # Calculate the percentage of the reward 1 in sample dataset
        rewardNum = len(np.where(batch_sample[:, self.n_features+1] == 1)[0])
        Rpercent_in_sample = (rewardNum/self.minibatch_size)*100
        print("--> reward 样本占: {0}%".format(Rpercent_in_sample))
        self.Rsample_plot.append(Rpercent_in_sample)

        q_summary, t_summary,q_next, predict_value = self.sess.run(
            [self.Qnetwork_merged, self.Tnetwork_merged, self.q_next, self.predict_value],
            feed_dict={
                self.s_: batch_sample[:, -self.n_features:],  # fixed params
                self.s: batch_sample[:, :self.n_features],  # newest params
            })
        self.train_writer.add_summary(q_summary, self.total_learning_step)
        self.train_writer.add_summary(t_summary, self.total_learning_step)

        # =======================================================
        # copy the matrix of predict_value to target_value.
        # bring the matrix of target value into correspondence with the one of predict_value
        # so that change the target value w.r.t predict_value
        target_value = predict_value.copy()

        # retrieve action (a) and the reward (r) of state (s) from the experience (samples)
        batch_sample_index =  np.arange(self.minibatch_size,dtype=np.int32)
        eval_act = batch_sample[:, self.n_features].astype(int)  # eval_act = [a0 a1 a2 ...]
        reward = batch_sample[:, self.n_features+1]  # reward = [r0 r1 r2 ... ]

        # check if there are any terminates(terminates or hells)
        win = np.where(batch_sample[:, self.n_features+1] == 1.0)[0]   # win the game, s_ is terminate
        fail = np.where(batch_sample[:, self.n_features+1] == -1.0)[0]  # lose the game, s_ is terminate
        terminates_index = np.sort(np.append(win, fail)) # the index of the terminates
        ter_act = eval_act[terminates_index]  # the corresponding action with terminate
        ter_reward = reward[terminates_index]  # the corresponding reward with terminate

        # delete the terminates' indices from all samples' indices
        if terminates_index.size != 0:
            batch_sample_index = np.delete(batch_sample_index, terminates_index)
            eval_act = np.delete(eval_act, terminates_index)
            reward = np.delete(reward, terminates_index)

        # retrieve the q_next state which is not terminate
        q_next_not_ter = q_next[batch_sample_index, :]

        # Calculate Q-Target Value
        # terminate
        if terminates_index.size != 0:
            target_value[terminates_index, ter_act] = ter_reward
        # NOT terminate
        target_value[batch_sample_index,eval_act] = reward + self.discountFactor * np.max(q_next_not_ter, axis=1)

        # Loss function and optimization -- begin to study (update the weights)
        loss_summary, opt, self.L = self.sess.run([self.loss_merged, self.optimizer, self.loss],
                               feed_dict={
                                   self.s: batch_sample[:, :self.n_features],
                                   self.target_value: target_value,
                               })
        self.train_writer.add_summary(loss_summary,self.total_learning_step)

        # increasing greedy 如果放在探索step里增加效果？？
        self.greedy_increment()
        if (self.total_learning_step >= 100) and (self.total_learning_step % 100 == 0):
            self.plot_Rp_memory()
            self.plot_Rp_sample()
        self.total_learning_step += 1

    def plot_Rp_memory(self):
        plt.figure()
        plt.plot(np.arange(len(self.Rmemory_plot)), self.Rmemory_plot)
        plt.title("Percentage of reward 1 in Memory")
        plt.ylabel('Reward')
        plt.xlabel('training steps')
        plt.show()


    def plot_Rp_sample(self):
        plt.figure()
        plt.plot(np.arange(len(self.Rsample_plot)), self.Rsample_plot)
        plt.title("Percentage of reward 1 in Sample")
        plt.ylabel('Reward')
        plt.xlabel('training steps')
        plt.show()


if __name__ == '__main__':
    test = DQN_st(4,2)








