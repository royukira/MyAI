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


class QBrainSimply:
    """
    For simply treasure game
    """

    def __init__(self,numState,ActionSet,greedy,learnRate,discountFactor,Max_episode):
        """initial Value"""
        self.numState = numState
        self.ActionSet = ActionSet
        self.greedy = greedy
        self.learnRate = learnRate
        self.discountFactor = discountFactor
        self.Max_episode = Max_episode

        """Initial Q Table"""
        self.Q_table = self.create_Q_table()

    def create_Q_table(self):
        matrix = np.zeros((self.numState, len(self.ActionSet)))
        table = pd.DataFrame(matrix,  # q table's value(&size)
                             columns=self.ActionSet  # Actions' name
                             )
        print(table)
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

    def train_brain(self):

        """Training part"""

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
            S = random.randint(0, self.numState-4)  # S不能等于treasure的position
            update_env(S, episode, step_count, self.numState)  # update the environment /
            """
            Repeat (For each step of an episode)
            """
            while not is_terminal:
                """
                Choose an action A
                """
                A = self.choose_action(state=S)
                print("\n[S,A] = [{0},{1}]".format(S,A))

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
                if S_Next is not "terminal":
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
                if S != "terminal":
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

            print(self.Q_table)
            time.sleep(1)

        return self.Q_table


class linear_Q(QBrainSimply):
    def __init__(self, numState,ActionSet,greedy,learnRate,discountFactor,Max_episode):
        super(linear_Q, self).__init__(numState,ActionSet,greedy,learnRate,discountFactor,Max_episode)
        """initial Value"""
        self.numState = numState
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

    def find_max_q(self, S_next):
        #TODO

        """
        Find the max Q value function of state S_next
        :param S_next:
        :return:
        """

        """
        Get the phi(S_next,A)
        """
        pending = {}
        index = S_next
        for i in self.ActionSet:
            if S_next == 0:
                x = self.X_S_A[:, index][:, np.newaxis]
            else:
                x = self.X_S_A[:, index+1][:, np.newaxis]
            y = np.dot(np.transpose(x), self.W)
            pending.setdefault(y,i)

        maxQ = max(pending)
        maxA = pending[maxQ]

        return maxQ,maxA

    def easy_find_max_q(self,S_next):
        if S_next == 0:
            w = self.W[0:len(self.ActionSet),:]
        else:
            w = self.W[S_next+1:S_next+1+len(self.ActionSet), :]

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
            s = random.randint(0, self.numState-4)  # S不能等于treasure的position
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
                    if a == 2:
                        x = self.X_S_A[:,s+1][:,np.newaxis]
                else:
                    x = self.X_S_A[:,s*a+1][:,np.newaxis]   # the x's shape is (#states * #action+1, 1)

                """
                Calculate the predict
                """
                q_predict = np.dot(np.transpose(x), self.W)

                """
                Calculate the target
                """
                if S_Next is "terminal":
                    self.target_error = R - q_predict
                else:
                    max_Q, max_A = self.easy_find_max_q(S_Next)
                    self.target_error = R + self.discountFactor * max_Q - q_predict

                """
                Update the parameters
                """
                self.W += self.learnRate * self.target_error  * x

                if S_Next is "terminal":
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


    # =========== Experience Replay ===============
    def create_memory(self, memory_size=20000):
        memory = np.zeros(memory_size, 4)
        return memory

    def store(self,e, memory, index):
        memory[index, :] = e
        return memory

    def collect_experience(self,iterNum=1000):

        """
        Import the environment
        """
        from training_env.Simply_Teasure_Game import update_env, get_env_feedback

        step_count = 0
        memory_size = 20000
        memory_count = 0
        memory = self.create_memory(memory_size)

        for i in range(iterNum):
            s = random.randint(0, self.numState - 4)  # S不能等于treasure的position
            update_env(s, i, step_count, self.numState)  # update the environment
            is_ter = False

            while is_ter is False:
                a = np.random.choice(self.ActionSet)

                S_Next, R = get_env_feedback(s, a, self.numState)

                experience = np.hstack((s, a, R, S_Next))

                index = memory_count % memory_size

                memory = self.store(experience, memory, index)

                memory_count += 1

        return memory, memory_count












