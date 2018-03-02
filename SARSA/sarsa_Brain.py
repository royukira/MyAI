"""
SARSA BRAIN

A on-policy learning

Difference between Q-Learning: Update way -- R + discount_factor * Q(s_next,a_next), where a_next is the next actual
                                             action

Move more carefully than Q-Learning

Implementation: Because the policy and the algorithm is similar to Q-learning
                so we can inherit Q_Barin_Maze.Q_Brain
"""

import numpy as np
import pandas as pd
from Q_learning.Q_Brain_Maze import Q_Brain


class sarsa(Q_Brain):
    def __init__(self,action,lr, df, greedy, episode):
        """
        Initial SARSA and inherit Q_Brain
        :param action: Action set
        :param lr: Learning Rate
        :param df: Discount Factor
        :param greedy: greedy value
        :param episode: num of episode
        """
        super(sarsa,self).__init__(action,lr, df, greedy, episode)
        self.actionSet = action
        self.learningRate = lr
        self.discountFactor = df
        self.greedy = greedy
        self.episode = episode
        self.brain = self.create_brain()
        self.s = None
        self.a = None
        self.next_a = None
        self.next_s = None

    def training(self, env):
        """
        Training the brain
        :param env: the environment
        :return: None
        """
        """
        Initial Record
        """
        # initial_record() # TODO

        """
        Repeat the episode
        """
        for e in range(self.episode):
            step_count = 0  # count the #episode
            is_done = False  # ending signal
            self.s = env.reset_exp()
            self.new_state(self.s)

            while True:
                """
                Fresh the env
                """
                env.render()

                """
                Train part
                """
                if step_count == 0:
                    self.a = self.choose_action(self.s)  # select the action

                q_predict = self.brain.loc[str(self.s), self.a]  # the old q of (s,a)

                self.next_s = env.update_env(self.a)  # move the explorer according to the action
                # and get the coordinate of the next state
                self.new_state(self.next_s)

                reward, is_done, _next_s = env.feedback()  # get the feedback of the action
                # (i.e. according to the next state, feedback the reward)

                
                # The next_a is the actual action of the next state
                # This is the difference
                # If the next state is Terminal or Hell, there is no necessary to select the next action
                if is_done is False:
                    self.next_a = self.choose_action(self.next_s)  # h和s状态下的policy是同一个
                    next_s_a = self.brain.loc[str(self.next_s), self.next_a]
                    q_target = reward + self.discountFactor * next_s_a
                else:
                    q_target = reward

                dis = q_target - q_predict

                # New value of (s,a)
                self.brain.loc[str(self.s), self.a] += self.learningRate * dis
                new_Q = self.brain.loc[str(self.s), self.a]

                # s <- next_s
                old_s = self.s
                self.s = self.next_s
                self.a = self.next_a
                step_count += 1

                """
                Hell or terminal
                """
                if is_done:
                    break
        # end of game
        print('game over')
        env.destroy()