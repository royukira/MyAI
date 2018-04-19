"""
A toy model for the experiment of analysis of memory effects

Action: a = {+1, -1}
State: s' = s + a
Reward: r = beta1 * (s + a) + bate2
Optimal Q-value: Q(s,a; beta) = r = beta1 * (s + a) + beta2;    # beta2 is more like a bias
Expected Q-value: Q_hat(s,a; theta) = theta1 * (s + a) + theta2;
TD-error:  e = [r(s,a; beta) + gamma * maxQ(s',a'; theta) - Q(s,a theta)]

"""
from Q_learning.Q_Brain_Simply import Memory as ER_memory
from Q_learning.QpER import Memory as pER_memory
from training_env import searchLinear_env as sle
from matplotlib import pyplot as plt
import numpy as np


def greedy_increment(greedy, total_learning_step, incrementRate=0.00004, greedy_max=0.9):
    """
    self-increment greeedy : epsilon += incrementRate / (epsilon + c)^2

    :param incrementRate: global increment rate  /  default: 0.001
    :param greedy_max: the max greedy  /  default: 0.95
    :return: None
    """
    if total_learning_step == 0:
        greedy = 0.1
    elif greedy < greedy_max:
        greedy += incrementRate / np.square(greedy + 0.00001)  # 0.00001 ensures the denominator will not be 0
    else:
        greedy = greedy_max
    return greedy


def InitMemory(memorySize,mode):
    memory = None
    if mode == "ER":
        memory = ER_memory(memorySize)
    elif mode == "pER":
        memory = pER_memory(memorySize)
    return memory


def runAgent(batchSize, memorySize, RL, gamma, NumStates=10, mode="ER"):
    """
    The agent start to play and learn
    :param batchSize: the size of mini-batch
    :param memorySize: the size of memory
    :param RL: the RL algorithm instance variable
    :param gamma: the discount factor
    :param NumStates: the number of states
    :param mode: the experience replay mode -- ER or pER
    :return:
    """

    # Initialize
    Cstep = 5  # learning transitions extracted from memory every C steps
    greedy = greedy_increment(0,0)  # the greedy with monotonic increasing
    maxEpisode = 20
    total_steps = 0
    env = sle.searchLin_env(maxState=NumStates)
    steps = []
    episodes = []
    dsteps = []
    Dist1 = []  # the distance of the target parameters and the parameters during the learning
    Dist2 = []

    # Initialize Memory

    memory = InitMemory(memorySize, mode)

    for i_episode in range(maxEpisode):

        s = 0  # every episode starts from s=0
        while True:
            epsilon = np.random.uniform()

            # Behavior Policy: epsilon-greedy policy
            if epsilon < greedy:
                a = RL.maxQvalue(s)  # TODO: realized by Q_Brain_linearSearch.py
            else:
                a = np.random.choice(RL.ActionSet)

            s_, r = env.get_feedback(s, a)
            transition = np.hstack((s, a, s_, r))

            if mode == "ER":
                memory.store_exp(transition)
            elif mode == "pER":
                memory.store(transition)

            learningSwitch =(total_steps % Cstep) == 0  # decide whether start to learn from memory: dTrue of False

            if (total_steps > memorySize) and learningSwitch:
                if mode == "ER":
                    print("--> Step {0} : Q-Learning with ER...".format(total_steps))
                    dist1, dist2 = RL.ERtrain(batchSize, gamma)  # TODO: realized by Q_Brain_linearSearch.py
                    Dist1.append(dist1)
                    Dist2.append(dist2)
                    dsteps.append(total_steps)

                elif mode == "pER":
                    print("--> Step {0} : Q-Learning with prioritized ER...".format(total_steps))
                    dist1, dist2 = RL.pERtrain(batchSize, gamma)  # TODO: realized by Q_Brain_linearSearch.py
                    Dist1.append(dist1)
                    Dist2.append(dist2)
                    dsteps.append(total_steps)

            if s_ == -1:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break
            else:
                s = s_

            total_steps += 1
            greedy = greedy_increment(greedy, total_steps)

    epi_step = np.vstack((episodes, steps))
    dist1_step = np.vstack((dsteps, Dist1))
    dist2_step = np.vstack((dsteps, Dist2))
    return epi_step, dist1_step, dist2_step


if __name__ == '__main__':

    color = ['r', 'b', 'c', 'g']

    # TODO: Initialize; the parameters below will be set after finishing the Q_Brain_linearSearch.py
    batchSize = None
    memorySize = None
    RL = None
    gamma = None
    NumStates = 10

    step, ER_dist1, ER_dist2 = runAgent(batchSize, memorySize, RL, gamma, NumStates, mode="ER")
    step2, pER_dist1, pER_dist2 = runAgent(batchSize, memorySize, RL, gamma, NumStates, mode="pER")

    # Plot total training step comparison
    plt.figure(1)
    plt.plot(step[0, :], step[1, :], c=color[0], label='Linear Q-learning with ER')
    plt.plot(step2[0, :], step2[1, :], c=color[1], label='Linear Q-learning with pER')
    plt.legend(loc='best')
    plt.ylabel('Total training step')
    plt.xlabel('Episode')
    plt.grid()
    plt.show()

    # Plot comparison of distances of ER
    plt.figure(2)
    plt.plot(ER_dist1[0, :], ER_dist1[1, :], c='r', label='Distance of theta1')
    plt.plot(ER_dist2[0, :], ER_dist2[1, :], c='b', label='Distance of theta2')
    plt.title('Analysis of Experience Replay')
    plt.legend(loc='best')
    plt.ylabel('Distance')
    plt.xlabel('Learning steps')
    plt.grid()
    plt.show()

    # Plot comparison of distances of pER
    plt.figure(3)
    plt.plot(pER_dist1[0, :], pER_dist1[1, :], c='r', label='Distance of theta1')
    plt.plot(pER_dist2[0, :], pER_dist2[1, :], c='b', label='Distance of theta2')
    plt.title('Analysis of prioritized Experience Replay')
    plt.legend(loc='best')
    plt.ylabel('Distance')
    plt.xlabel('Learning steps')
    plt.grid()
    plt.show()