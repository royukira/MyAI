"""
A toy model for the experiment of analysis of memory effects

Action: a = {+1, -1}
State: s' = s + a
Reward: r = beta1 * (s + a) + bate2
Optimal Q-value: Q(s,a; beta) = r = beta1 * (s + a) + beta2;    # beta2 is more like a bias
Expected Q-value: Q_hat(s,a; theta) = theta1 * (s + a) + theta2;
TD-error:  e = [r(s,a; beta) + gamma * maxQ(s',a'; theta) - Q(s,a theta)]

"""
class searchLin_env:
    def __init__(self, maxState=6, beta1=2, beta2=0.1):

        self.terminate = maxState + 1
        # e.g. if there are 6 states in the 1D search space, the 7th state is the terminate that end the episode

        self.beta1 = beta1
        self.beta2 = beta2

    def nextState(self, s, a):
        if s+a < self.terminate:
            return s + a
        else:
            return -1

    def getReward(self, s, a):
        beta1 = 2
        beta2 = 0.1  # beta2 is more like a bias
        reward = beta1 * (s + a) + beta2
        return reward

    def get_feedback(self, s, a):
        """
        combine nextState and getReward
        :param: s: the current state
        :param: a: the action
        :return: s_ = nextState(s,a);  r = getReward(s,a)
        """

        s_ = self.nextState(s, a)
        r = self.getReward(s, a)

        return s_, r







