"""
MyAI Experience

This is Q_learning package

Reference: https://morvanzhou.github.io/tutorials
"""
import numpy as np
import pandas as pd
import random
import time
import os


class QBrainSimply:

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
        if not os.path.exists("./Log/%s_QBrainSimply_log.txt" % now_time):
            f = open("./Log/%s_QBrainSimply_log.txt" % now_time,'w')
            f.close()

        """
        Repeat the episode until S gets to the rightmost position (i.e. get the treasure)
        """
        for episode in range(self.Max_episode):  # training episode; a episode from initial S(i.e. State) to terminal S
            step_count = 0  # count the #episode
            is_terminal = False  # ending signal
            S = random.randint(0, self.numState-2)  # S不能等于5
            update_env(S, episode, step_count, self.numState)  # update the environment / TODO

            while not is_terminal:
                A = self.choose_action(state=S)
                S_Next, R = get_env_feedback(S,A,self.numState)  # get the feedback from the environment/ TODO
                q_predict = self.Q_table.loc[S,A]  # 估计的价值

                if S_Next is not "terminal":
                    q_target = R + self.discountFactor * self.Q_table.iloc[S_Next,:].max()  # 实际的价值
                else:
                    q_target = R  # 实际的价值
                    is_terminal = True

                q_dis = q_target - q_predict
                self.Q_table.loc[S,A] += self.learnRate * q_dis  # [S,A] in Q table is updated
                S = S_Next
                step_count += 1
                interaction = update_env(S, episode, step_count,self.numState)

                """
                Record Learning Process
                """
                if S != "terminal":
                    train_log = open("./Log/%s_QBrainSimply_log.txt" % now_time, 'a+')
                    train_log.write("Update to " + str(interaction))
                    train_log.write("\n--> Episode: %s\
                                    \n--> Step: %s \
                                    \n--> Next State: %s \
                                    \n--> Chosen Action by current state: %s \
                                    \n--> MAX[S_next,A] = %.7f\
                                    \n--> Reward: %d \
                                    \n--> Target-Predict: %.7f \
                                    \n--> Q_target: %.7f \
                                    \n--> Q_predict: %.7f \
                                    \n===========\n"
                                    % (episode,step_count, S, str(A), self.Q_table.iloc[S,:].max(), R, q_dis, q_target, q_predict))
                    train_log.close()

            print(self.Q_table)
            time.sleep(1)

        return self.Q_table







