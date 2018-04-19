"""
A toy model for the experiment of analysis of memory effects

Action: a = {0,1}
State: s' = s + a
(approximated) Reward: r = w * s * a + b
(real) Reward: r = w' * s * a + b'; where w' and b' are objective parameters
Q-value: Q(s,a;w,b) = (w * s * a + b) + gamma * max(w * s' * a' + b )
TD-error:  e = [r(s,a;w',b') + gamma * maxQ(s',a';w,b) - Q(s,a;w,b)]
param-error: we = w - w'; be = b - b'、

"""


def nextState(s, a):
    return s+a


def getReward(s, a, w, b):
    return (w * s * a + b)




