"""
My AI experience: Maze Game

This script is about training a machine through Q-learning

Return: The machine's Q brain (Q table)

"""

import numpy as np
import pandas as pd



class Q_Brain:
    def __init__(self, action,lr, df, greedy, episode):
        """
        Initial Q_Brain
        :param action: Action set
        :param lr: Learning Rate
        :param df: Discount Factor
        :param greedy: greedy value
        :param episode: num of episode
        """
        self.actionSet = action
        self.learningRate = lr
        self.discountFactor = df
        self.greedy = greedy
        self.episode = episode
        self.brain = self.create_brain()

    def create_brain(self):
        """
        Create the new brain (table)
        :return: brain
        """
        brain = pd.DataFrame(columns=self.actionSet,dtype=np.float64)
        return brain

    def new_state(self,state):
        """
        Check the state if it has already been in the brain
        Otherwise, append the new state
        :param state: The current state (e.g. the state of explorer: [x0,y0,x1,y1])
        :return: None
        """
        if str(state) not in self.brain.index:
            new_state = pd.Series(data=[0]*len(self.actionSet),  # initial value 0 to new state
                                index=self.actionSet,  # the index is the columns name of the brain
                                name=str(state))        # the name is the index name of the brain

            """
            append the new state
            """
            self.brain = self.brain.append(new_state) # 一定要以赋值形式返回Q table（self.brain）
            print(self.brain)

    def choose_action(self, state):
        """
        Choose the action
        if smaller than greedy value, choose the action with highest value
        otherwise, choose an action randomly
        :return: action
        """
        self.new_state(state)
        state_action = self.brain.loc[str(state),:]
        # loc: gets rows (or columns) with particular labels from the index
        # iloc: gets rows (or columns) at particular positions in the index (so it only takes integers).
        # ix: usually tries to behave like loc but falls back to behaving like iloc if a label is not present in the index.
        state_action = state_action.reindex(np.random.permutation(state_action.index))
        # disrupt the order. in case always
        # choose the same action when the
        # action values are the same (e.g. all 0)

        # state_action = pd.Series(data=state_action, index=self.actionSet,dtype=np.float64)
        # in case the dtype of state_action is object which cannot use idxmax() or argmax() function
        # idxmax: Return index of first occurrence of maximum over requested axis.


        gv = np.random.uniform()

        if gv <= self.greedy:
            action = state_action.idxmax()  # choose the action with the max value
            print("--> Max Choice: %s" % action)
            print(state_action)
        else:
            action = np.random.choice(self.actionSet)
            print("--> Random choice: %s" % action)

        return action

    def training(self, env):
        """
        Training the brain
        :param env: the environment

        """
        """
        Initial Record
        """
        #initial_record() # TODO

        """
        Repeat the episode
        """
        for e in range(self.episode):

            step_count = 0  # count the #episode
            is_done = False  # ending signal
            env.reset_exp()
            s = env.exp_coord
            self.new_state(s)

            while True:
                """
                Fresh the env
                """
                env.render()

                """
                Train part
                """
                a = self.choose_action(s)  # select the action

                q_predict = self.brain.loc[str(s),a]  # the old q of (s,a)

                next_s = env.update_env(a)  # move the explorer according to the action
                # and get the coordinate of the next state
                self.new_state(next_s)

                reward, is_done, _next_s = env.feedback() # get the feedback of the action
                # (i.e. according to the next state, feedback the reward)

                if is_done is False :
                    next_s_max = self.brain.loc[str(next_s),:].max()
                    q_target = reward + self.discountFactor * next_s_max
                else:
                    q_target = reward

                dis = q_target - q_predict

                # New value of (s,a)
                self.brain.loc[str(s),a] += self.learningRate * dis
                new_Q = self.brain.loc[str(s), a]

                # s <- next_s
                old_s = s
                s = next_s
                step_count += 1

                """
                Hell or terminal
                """
                if is_done:
                    break
        # end of game
        print('game over')
        env.destroy()








