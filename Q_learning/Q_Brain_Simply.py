"""
MyAI Experience

This is Q_learning simply version package

Reference: https://morvanzhou.github.io/tutorials
"""
import numpy as np
import pandas as pd
import random
import time
import os
import copy
import pymysql as pms


class QBrainSimply:
    """
    For simply treasure game
    """

    def __init__(self,numState,ActionSet,greedy,learnRate,discountFactor,Max_episode):
        """initial Value"""
        self.numState = numState - 1
        self.ActionSet = ActionSet
        self.greedy = greedy
        self.learnRate = learnRate
        self.discountFactor = discountFactor
        self.Max_episode = Max_episode

        """Initial Q Table"""
        self.Q_table = None

    def create_Q_table(self, mode):
        if mode == "uniform":
            matrix = np.random.uniform(size=(self.numState, len(self.ActionSet)))
            table = pd.DataFrame(matrix,  # q table's value(&size)
                                 columns=self.ActionSet  # Actions' name
                                 )
        elif mode == "zero":
            matrix = np.zeros((self.numState, len(self.ActionSet)))
            table = pd.DataFrame(matrix,  # q table's value(&size)
                                 columns=self.ActionSet  # Actions' name
                                 )
        #print(table)
        return table  # return Q-Table

    def choose_action(self,state):
        state_action = self.Q_table.iloc[state,:]
        random_choose = np.random.uniform()  # 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high
                                                # low default: 0 ; high default: 1

        """Act non-greedy action or all value of action is 0 -- random choose action"""
        if (random_choose > self.greedy) or (state_action.all() == 0):
            action = np.random.choice(self.ActionSet)
            return action

        """Act greedy acction """
        if (random_choose < self.greedy) and (state_action.all() != 0):
            action = state_action.argmax()
            return action

    def test_choose_action(self, table, s):
        state_action = table.iloc[s,:]
        action = state_action.argmax()
        return action

    def train_brain(self,write=False):

        """Training part"""
        """
        Initial Q table
        """
        self.Q_table = self.create_Q_table("zero")
        """
        Import the environment
        """
        from training_env.Simply_Teasure_Game import update_env, get_env_feedback

        """
        Create the learning log
        """
        now_time = time.strftime("%Y-%m-%d", time.localtime())
        if not os.path.exists("../Log/%s_QBrainSimply_log.txt" % now_time):
            f = open("../Log/%s_QBrainSimply_log.txt" % now_time,'w')
            f.close()

        """
        Repeat the episode until S gets to the rightmost position (i.e. get the treasure)
        """
        for episode in range(self.Max_episode):  # training episode; a episode from initial S(i.e. State) to terminal S
            """
            Initial S
            """
            step_count = 0  # count the #episode
            is_terminal = False  # ending signal
            S = 0 # S不能等于treasure的position
            update_env(S, episode, step_count, self.numState)  # update the environment /
            """
            Repeat (For each step of an episode)
            """
            while not is_terminal:
                """
                Choose an action A
                """
                A = self.choose_action(state=S)
                #print("\n[S,A] = [{0},{1}]".format(S,A))

                """
                Get feedback of action A with State S from the environment
                Return next state and reward (i.e observe the next state and reward)
                """
                S_Next, R = get_env_feedback(S,A,self.numState)  # get the feedback from the environment/

                """
                Get the predict value (i.e old value of Q[S,A]) from Q table
                """
                q_predict = self.Q_table.loc[S,A]  # 估计的价值

                """
                Calculate the real value (q_target)
                """
                if S_Next != -1 :
                    q_target = R + self.discountFactor * self.Q_table.iloc[S_Next,:].max()  # 实际的价值
                else:
                    q_target = R  # 实际的价值,terminal的action都为0
                    is_terminal = True

                q_dis = q_target - q_predict

                """
                New Q[S,A] is updated
                """
                self.Q_table.loc[S,A] += self.learnRate * q_dis  # [S,A] in Q table is updated
                new_Q = self.Q_table.loc[S,A]
                old_S = S
                S = S_Next
                step_count += 1
                interaction = update_env(S, episode, step_count,self.numState)

                """
                Record Learning Process
                """
                if S != -1 and write == True:
                    train_log = open("../Log/%s_QBrainSimply_log.txt" % now_time, 'a+')
                    train_log.write("Update to " + str(interaction))
                    train_log.write("\n--> Episode: %s\
                                    \n--> Step: %s \
                                    \n--> Next State: %s \
                                    \n--> Chosen Action by current state: %s \
                                    \n--> New Q[%d,%s] = %.7f\
                                    \n--> MAX Q[S_next,A] = %.7f\
                                    \n--> Reward: %d \
                                    \n--> Target-Predict: %.7f \
                                    \n--> Q_target: %.7f \
                                    \n--> Q_predict: %.7f \
                                    \n===========\n"
                                    % (episode,
                                       step_count,
                                       S,
                                       str(A),
                                       old_S, str(A),
                                       new_Q,
                                       self.Q_table.iloc[S,:].max(),
                                       R,
                                       q_dis,
                                       q_target,
                                       q_predict))
                    train_log.close()
                else:
                    train_log = open("../Log/%s_QBrainSimply_log.txt" % now_time, 'a+')
                    train_log.write("Update to Terminal")
                    train_log.write("\n--> Episode: %s\
                                                        \n--> Step: %s \
                                                        \n--> Chosen Action by current state: %s \
                                                        \n--> Reward: %d \
                                                        \n--> New Q[%s,%s] = %.7f\
                                                        \n--> Target-Predict: %.7f \
                                                        \n--> Q_target: %.7f \
                                                        \n--> Q_predict: %.7f \
                                                        \n===========\n"
                                    % (
                                    episode,
                                    step_count,
                                    str(A),
                                    R,
                                    str(old_S), str(A),
                                    new_Q,
                                    q_dis,
                                    q_target,
                                    q_predict))
                    train_log.close()
            print("\n")
            #print(self.Q_table)
            #time.sleep(0.1)

        return self.Q_table

    def batch_table_train(self, memory, batchSize, max_epoch):
        """Training part"""
        """
        Initial Q table
        """
        self.Q_table = self.create_Q_table("uniform")

        from Utility_tool.laplotter import LossAccPlotter
        """
        Repeat the episode until s gets to the rightmost position (i.e. get the treasure)
        """
        total_epoch = 1  # count the #episode
        is_terminal = False  # ending signal
        save_path = "/Users/roy/Documents/GitHub/MyAI/Log/BCW_loss/{0}_state_table.png".format(self.numState)
        plotter = LossAccPlotter(title="Loss of Tabular Look-up Table with {0} states".format(self.numState),
                                 save_to_filepath=None,
                                 show_acc_plot=False,
                                 show_plot_window=False,
                                 show_regressions=False,
                                 LearnType="table")

        error_buffer = copy.deepcopy(self.Q_table)
        while is_terminal is False:  # training episode; a episode from initial s(i.e. State) to terminal s
            """
            Initial s
            """
            print("--> Epoch {0} Training... \n".format(total_epoch))
            batchIndex = np.random.choice(memory.shape[0], size = batchSize)
            batchSample = memory[batchIndex,:]
            error = 0

            for sample in batchSample:
                s = int(sample[0])
                a = int(sample[1])
                r = int(sample[2])
                s_ = int(sample[3])

                q_predict = self.Q_table.loc[s, a]

                """
                Calculate the real value (q_target)
                """
                if s_ != -1:
                    q_target = r + self.discountFactor * self.Q_table.iloc[s_, :].max()  # 实际的价值
                else:
                    q_target = r  # 实际的价值,terminal的action都为0

                q_dis = q_target - q_predict

                error_buffer.loc[s, a] += self.learnRate * q_dis
                error += np.square(q_dis)

            self.Q_table = copy.deepcopy(error_buffer)

            mse = np.sqrt(error)/batchSize
            print("--> Epoch {0}'s error: {1}\n".format(total_epoch, mse))
            print("==============================================\n")

            plotter.add_values(total_epoch, loss_train=mse)

            """
            if mse < 0.0001:
                is_terminal = True
            """
            if total_epoch > max_epoch:
                is_terminal = True
            else:
                total_epoch += 1

        print("--> Total learning step: {0}".format(total_epoch))
        plotter.save_plot(save_path)
        plotter.block()

        return self.Q_table

    def test_policy(self, table, episode):

        from training_env.Simply_Teasure_Game import update_env, get_env_feedback

        for episode in range(episode):  # training episode; a episode from initial s(i.e. State) to terminal s
            """
            Initial s
            """
            step_count = 0  # count the #episode
            is_terminal = False  # ending signal
            s = 0
            update_env(s, episode, step_count, self.numState)  # update the environment
            while is_terminal is False:
                """
                Choose an action a
                """
                a = self.test_choose_action(table, s)
                # print("\n[s,a] = [{0},{1}]".format(s, a))

                """
                Get feedback of action a with State s from the environment
                Return next state and reward (i.e observe the next state and reward)
                """
                S_Next, R = get_env_feedback(s, a, self.numState)  # get the feedback from the environment/

                if S_Next == -1:
                    is_terminal = True
                else:
                    s = S_Next
                    step_count += 1
                interaction = update_env(S_Next, episode, step_count, self.numState)




class linear_Q(QBrainSimply):

    def __init__(self, numState,ActionSet,greedy,learnRate,discountFactor,Max_episode=None):
        super(linear_Q, self).__init__(numState,ActionSet,greedy,learnRate,discountFactor,Max_episode)
        """initial Value"""
        self.numState = numState - 1  # 减去terminal
        self.ActionSet = ActionSet
        self.greedy = greedy
        self.learnRate = learnRate
        self.discountFactor = discountFactor
        self.Max_episode = Max_episode
        self.W = np.random.uniform(0,0.1,size=self.numState*len(self.ActionSet)+1)[:,np.newaxis]  # 初始化W (#states * #action+1, 1)
                                                                                                  # 加1是bias
        self.X_S_A = np.identity(self.numState*len(self.ActionSet)+1) # #states * I(#action+1 , #states * #action+1)
        self.target_error = 0
        """ parameter matrix """
        self.para_matrix = self.create_para_matrix()

    def create_para_matrix(self):
        para_matrix = pd.DataFrame(dtype=np.float64)
        return para_matrix

    def new_X(self, X, W):
        if str(X) not in self.para_matrix.index:
            new_state = pd.Series(data= W,
                                name=str(X))        # the name is the index name of the brain

            """
            append the new state
            """
            self.para_matrix = self.para_matrix.append(new_state) # 一定要以赋值形式返回Q table（self.brain）
            #print(self.para_matrix)

    def choose_action(self,state):
        random_choose = np.random.uniform()  # 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high
        # low default: 0 ; high default: 1

        """Act non-greedy action or all value of action is 0 -- random choose action"""
        if (random_choose > self.greedy):
            action = np.random.choice(self.ActionSet)
            return action

        """Act greedy acction """
        if (random_choose < self.greedy):
            q, action = self.easy_find_max_q(state)
            return action

    def test_choose_action(self, w, S_next):
        if S_next == 0:
            w_ = w[0:len(self.ActionSet),:]
        else:
            w_ = w[S_next * len(self.ActionSet):S_next * len(self.ActionSet) + 2, :]

        maxQ = w_.max()
        maxA = np.where(w_ == maxQ)[0][0]
        maxA += 1

        return maxA

    def easy_find_max_q(self,S_next):
        if S_next == 0:
            w = self.W[0:len(self.ActionSet),:]
        else:
            w = self.W[S_next*len(self.ActionSet):S_next*len(self.ActionSet)+2, :]

        maxQ = w.max()
        maxA = np.where(w == maxQ)[0][0]
        maxA += 1

        return maxQ, maxA

    def linear_train(self):
        """Training part"""

        """
        Import the environment
        """
        from training_env.Simply_Teasure_Game import update_env, get_env_feedback


        """
        Repeat the episode until s gets to the rightmost position (i.e. get the treasure)
        """
        total_step_count = 0  # count the #episode
        for episode in range(self.Max_episode):  # training episode; a episode from initial s(i.e. State) to terminal s
            """
            Initial s
            """
            step_count = 0  # count the #episode
            is_terminal = False  # ending signal
            s = 0
            update_env(s, episode, step_count, self.numState)  # update the environment

            while is_terminal is False:
                """
                Choose an action a
                """
                a = self.choose_action(state=s)
                #print("\n[s,a] = [{0},{1}]".format(s, a))

                """
                Get feedback of action a with State s from the environment
                Return next state and reward (i.e observe the next state and reward)
                """
                S_Next, R = get_env_feedback(s, a, self.numState)  # get the feedback from the environment/

                """
                Get the phi(s,a) from BIG_phi(s,a) i.e. self.X_S_A
                """
                if s == 0:
                    if a == 1:
                        x = self.X_S_A[:,s][:,np.newaxis]
                    elif a == 2:
                        x = self.X_S_A[:,s+1][:,np.newaxis]
                else:
                    if a == 1:
                        x = self.X_S_A[:, s * len(self.ActionSet)][:, np.newaxis]  # the x's shape is (#states * #action+1, 1)
                    elif a == 2:
                        x = self.X_S_A[:, s * len(self.ActionSet) + 1][:, np.newaxis]  # the x's shape is (#states * #action+1, 1)

                """
                Calculate the predict
                """
                q_predict = np.dot(np.transpose(x), self.W)

                """
                Calculate the target
                """
                if S_Next == -1:
                    self.target_error = R - q_predict
                else:
                    max_Q, max_A = self.easy_find_max_q(S_Next)
                    self.target_error = R + self.discountFactor * max_Q - q_predict

                """
                Update the parameters
                """
                self.W += self.learnRate * self.target_error  * x

                if S_Next == -1:
                    is_terminal = True
                else:
                    s = S_Next
                    step_count += 1
                    total_step_count += 1
                interaction = update_env(S_Next, episode, step_count, self.numState)

            if self.target_error  < 0.00001:
                print("\nTraining finish")
                break
        print("--> Total learning step: {0}".format(total_step_count))

        return self.W

    def batch_linear_train(self, memory, batchSize):
        """Training part"""

        from Utility_tool.laplotter import LossAccPlotter
        """
        Repeat the episode until s gets to the rightmost position (i.e. get the treasure)
        """
        #save_path = "/Users/roy/Documents/GitHub/MyAI/Log/BCW_loss/{0}_state_LFA.png".format(self.numState)
        """
        plotter = LossAccPlotter(title="Loss of Linear FA with {0} states".format(self.numState),
                                 save_to_filepath=None,
                                 show_acc_plot=False,
                                 show_plot_window=False,
                                 show_regressions=False,
                                 LearnType="LFA"
                                 )
        """

        batchIndex = np.random.choice(memory.shape[0], size=batchSize)
        batchSample = memory[batchIndex, :]
        w_increment = np.zeros((self.numState * len(self.ActionSet) + 1, 1))

        for sample in batchSample:
            x = 0
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

            w_increment += self.target_error * x

        """
        update 
        """
        self.W += self.learnRate * w_increment

        gradient = np.linalg.norm(w_increment) / batchSize
        print("--> Linear Q-learning's gradient: {0}\n".format(gradient))
        print("==============================================\n")

        return self.W, gradient

    def test_policy(self, w, episode):

        from training_env.Simply_Teasure_Game import update_env, get_env_feedback

        for episode in range(episode):  # training episode; a episode from initial s(i.e. State) to terminal s
            """
            Initial s
            """
            step_count = 0  # count the #episode
            is_terminal = False  # ending signal
            s = 0
            update_env(s, episode, step_count, self.numState)  # update the environment
            while is_terminal is False:
                """
                Choose an action a
                """
                a = self.test_choose_action(w=w, S_next=s)
                # print("\n[s,a] = [{0},{1}]".format(s, a))

                """
                Get feedback of action a with State s from the environment
                Return next state and reward (i.e observe the next state and reward)
                """
                S_Next, R = get_env_feedback(s, a, self.numState)  # get the feedback from the environment/

                if S_Next == -1:
                    is_terminal = True
                else:
                    s = S_Next
                    step_count += 1
                interaction = update_env(S_Next, episode, step_count, self.numState)

    # =========== Experience Replay ===============

    def create_memory(self, memory_size=2000):
        memory = np.zeros((memory_size, 4))
        return memory

    def store(self,e, memory, index):
        memory[index, :] = e
        return memory

    def collect_experience(self,iterNum=1000, threadID=None):

        """
        Import the environment
        """
        from training_env.Simply_Teasure_Game import update_env, get_env_feedback

        step_count = 0
        memory_size = 2000  #np.power(2, self.numState+1) - 2
        print("\n---> {0}: memory size: {1}".format(threadID,memory_size))
        memory_count = 0
        memory = self.create_memory(memory_size)
        check_dist = {}

        for i in range(iterNum):
            print("\n--->{0}: iter: {1}".format(threadID,i))
            s = random.randint(0, self.numState-1)  # S不能等于treasure的position
            update_env(s, i, step_count, self.numState)  # update the environment
            is_ter = False

            while is_ter is False:
                a = np.random.choice(self.ActionSet)

                S_Next, R = get_env_feedback(s, a, self.numState)

                exp_tuple = (s, a, R, S_Next)

                experience = np.hstack(exp_tuple)

                interaction = update_env(S_Next, i, step_count, self.numState)

                if exp_tuple in check_dist:
                    pass
                else:
                    index = memory_count % memory_size

                    memory = self.store(experience, memory, index)

                    memory_count += 1

                    check_dist.setdefault(exp_tuple,0)

                if S_Next == -1:
                    is_ter = True
                else:
                    s = S_Next
        print("\n---> {0}: Total {1} transitions".format(threadID,len(check_dist)))

        actual_memory = memory[:len(check_dist),:]  # 记忆矩阵中后面全0的都不要
        return actual_memory, memory_count




class oracle_Q(linear_Q):

    def __init__(self, numState, ActionSet, greedy, learnRate, discountFactor, Max_episode):
        super(oracle_Q, self).__init__(numState, ActionSet, greedy, learnRate, discountFactor, Max_episode)
        """initial Value"""
        self.numState = numState - 1  # 减去terminal
        self.ActionSet = ActionSet
        self.greedy = greedy
        self.learnRate = learnRate
        self.discountFactor = discountFactor

        self.db = self.connectSQL()
        self.cursor = self.db.cursor()


    def connectSQL(self):
        db = pms.Connect("localhost", "root", "Zhang715", "BCW")
        return db


    def createPriorityTable(self, tableName):
        sql = "CREATE TABLE {0}(\
                STATE_ SMALLINT(5),\
                ACTION_ SMALLINT(5),\
                REWARD_ SMALLINT(5),\
                STATE_NEXT SMALLINT(5),\
                ERROR FLOAT )".format(tableName)
        try:
            self.cursor.execute(sql)
            self.db.commit()
            print("Already Create TABLE PRIORITY")
        except:
            print("CANNOT CREATE TABLE PRIORITY")
            self.db.rollback()


    def insertMemory(self, memory, tableName):
        try:
            for m in memory:
                s = int(m[0])
                a = int(m[1])
                r = int(m[2])
                s_ = int(m[3])

                insert = "INSERT INTO %s(\
                          STATE_,\
                          ACTION_,\
                          REWARD_,\
                          STATE_NEXT,\
                          ERROR)\
                          VALUES ('%d','%d','%d','%d',5)" % (tableName,s, a, r, s_)
                try:
                    self.cursor.execute(insert)
                    print("Already insert {0}".format(m))
                except:
                    print("CANNOT INSERT {0}".format(m))
                    self.db.rollback()

            self.db.commit()
        except:
            self.db.rollback()


    def updateError(self,error, s, a, tableName):
        error = error[0][0]
        update = "UPDATE {0} SET ERROR = {1} WHERE STATE_ = {2} AND ACTION_ = {3}".format(tableName,error,s,a)

        try:
            self.cursor.execute(update)
            self.db.commit()
            print("Already Update ERROR of State:{0} ACTION:{1}".format(s, a))
        except:
            self.db.rollback()

    def selectBatch(self,batchSize,tableName):
        select = " SELECT STATE_, ACTION_, REWARD_, STATE_NEXT FROM {0} \
                   ORDER BY ERROR DESC ".format(tableName)

        try:
            self.cursor.execute(select)
            samples = self.cursor.fetchall()
            batchSample = samples[0:batchSize + 1]
            return batchSample  # Return a Tuple
        except:
            self.db.rollback()


    def oracle_train(self,batchSize, max_epoch, tableName):
        """Training part"""

        from Utility_tool.laplotter import LossAccPlotter
        """
        Repeat the episode until s gets to the rightmost position (i.e. get the treasure)
        """
        total_epoch = 1  # count the #episode
        is_terminal = False  # ending signal
        save_path = "/Users/roy/Documents/GitHub/MyAI/Log/BCW_loss/{0}_state_oracle.png".format(self.numState)
        plotter = LossAccPlotter(title="Loss of oracle memory with {0} states".format(self.numState),
                                 save_to_filepath=None,
                                 show_acc_plot=False,
                                 show_plot_window=False,
                                 show_regressions=False,
                                 LearnType="LFA"
                                )

        while is_terminal is False:

            print("--> Epoch {0} Training... \n".format(total_epoch))

            batchSample = np.array(self.selectBatch(batchSize, tableName))
            w_increment = np.zeros((self.numState * len(self.ActionSet) + 1, 1))

            for sample in batchSample:
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

                w_increment += self.target_error * x



                """
                Update the error of state stored in the oracle
                """
                self.updateError(np.absolute(self.target_error), s, a, tableName)

            """
            update 
            """
            self.W += self.learnRate * w_increment

            mse = np.linalg.norm(w_increment) / batchSize


            print("--> Epoch {0}'s error: {1}\n".format(total_epoch, mse))
            print("==============================================\n")

            plotter.add_values(total_epoch, loss_train=mse)

            """
            if error < 0.00001:
                is_terminal = True

            """
            if total_epoch % 300 == 0:
                save_path_100 = "/Users/roy/Documents/GitHub/MyAI/Log/BCW_loss/{0}_state_oracle.png".format(total_epoch)
                plotter.save_plot(save_path_100)
            if mse < 0.001:
                is_terminal = True
            if total_epoch > max_epoch:
                is_terminal = True
            else:
                total_epoch += 1

        print("--> Total learning step: {0}".format(total_epoch))
        plotter.save_plot(save_path)
        plotter.block()
        return self.W


class Memory:
    def __init__(self,memorysize):
        self.ms =memorysize
        self.memory = np.zeros((self.ms, 4))
        self.memory_counter = 0

    def store_exp(self,e):
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


if __name__ == '__main__':
    N_STATES = 26  # the length of the 1 dimensional world
    ACTIONS = [1, 2]  # available actions 1:right, 2:wrong
    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 1 - (1 / N_STATES - 1)  # discount factor
    MAX_EPISODES = 100  # maximum episodes

    test_db = oracle_Q(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                               discountFactor=GAMMA,
                               Max_episode=MAX_EPISODES)

    #test_db.createPriorityTable()
    memory = np.load("/Users/roy/Documents/GitHub/MyAI/Run/blind_cliffwalk_experience/memory_4.npy")
    #test_db.insertMemory(memory)









