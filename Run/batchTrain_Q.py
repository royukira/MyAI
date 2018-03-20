from Q_learning.Q_Brain_Simply import linear_Q
from Q_learning.Q_Brain_Simply import QBrainSimply
from Q_learning.Q_Brain_Simply import oracle_Q
from Q_learning.Q_Brain_Simply import Memory as my
from Q_learning.QForget import priority_train
import numpy as np
import threading
from training_env import Simply_Teasure_Game as stg
from matplotlib import pyplot as plt


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


def train(mode,threadID=None):
    """Setting"""
    N_STATES = 16 # the length of the 1 dimensional world
    ACTIONS = [1, 2]  # available actions 1:right, 2:wrong
    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 1 - (1 / N_STATES - 1)  # discount factor
    MAX_EPISODES = 100  # maximum episodes
    batchSize = 10
    max_epoch = 1000

    """
    Load the memory
    """
    memory = np.load("./blind_cliffwalk_experience/memory_15.npy")

    if mode == "LFA":
        train_model = linear_Q(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                               discountFactor=GAMMA,
                               Max_episode=MAX_EPISODES)

        train_w = train_model.batch_linear_train(memory=memory, batchSize=batchSize, max_epoch=max_epoch)

        print("--> Thread: {0} well trained...\n--> {0} Start to test...".format(threadID))

        train_model.test_policy(train_w, 15)

    if mode == "table":
        train = QBrainSimply(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                             discountFactor=GAMMA,
                             Max_episode=MAX_EPISODES)

        train_table = train.batch_table_train(memory, batchSize, max_epoch)

        print("--> Thread: {0} well trained...\n--> {0} Start to test...".format(threadID))

        train.test_policy(train_table, 15)

    if mode == "priority":
        tableName = "PRIORITY{0}".format(N_STATES-1)

        train = priority_train(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                             discountFactor=GAMMA,
                             Max_episode=MAX_EPISODES,
                               memorySize=memory.shape[0])

        train.load_memory(memory)
        train_w, step = train.priorityTrain(batchSize, max_epoch)

        train.plot(step)

        print("--> Thread: {0} well trained...\n--> {0} Start to test...".format(threadID))
        print(train_w)

        train.test_policy(train_w, 15)



    if mode == "oracle":
        tableName = "PRIORITY{0}".format(N_STATES-1)

        train = oracle_Q(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                             discountFactor=GAMMA,
                             Max_episode=MAX_EPISODES)

        train.createPriorityTable(tableName)
        train.insertMemory(memory, tableName)
        train_w = train.oracle_train(batchSize, max_epoch, tableName)

        print("--> Thread: {0} well trained...\n--> {0} Start to test...".format(threadID))

        train.test_policy(train_w, 15)

def train_with_play(batchSize,numState,RL,memory=None,mode=None):
    Cstep  = 5
    greedy = greedy_increment(0,0)
    total_steps = 0
    steps = []
    episodes = []
    awcMeans = []
    awcSteps = []
    memory = my(30)
    for i_episode in range(20):
        s=0
        stg.update_env(s, i_episode, total_steps, numState-1)
        while True:
            # env.render()
            random_choose = np.random.uniform()

            if random_choose < greedy:
                q,a = RL.easy_find_max_q(s)
            else:
                a = np.random.choice(RL.ActionSet)

            S_Next, R = stg.get_env_feedback(s, a, numState-1)

            transition = np.hstack((s, a, R, S_Next))
            if mode == "l":
                memory.store_exp(transition)
            if mode == "p":
                RL.memory.store(transition)

            is_learn = total_steps % Cstep

            if (total_steps > 200) and (is_learn == 0):
                if mode == "p":
                    print("--> Step {0} : Prioritized Learning...".format(total_steps))
                    awc_mean = RL.priorityTrain(batchSize)
                    awcMeans.append(awc_mean)
                    awcSteps.append(total_steps)
                if mode == "l":
                    print("--> Step {0} : Uniform Learning...".format(total_steps))
                    RL.batch_linear_train(memory.memory,batchSize)


            if R == 1:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            if S_Next == -1:
                s = 0
            else:
                s = S_Next
            #stg.update_env(s, i_episode, total_steps, numState)
            total_steps += 1
            greedy = greedy_increment(greedy,total_steps)

    epi_step = np.vstack((episodes, steps))
    awc_mean_step = np.vstack((awcSteps, awcMeans))
    return epi_step, awc_mean_step




class TrainThread (threading.Thread):
    def __init__(self, threadID, mode,batchSize,N_STATES,train):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.mode = mode
        self.batchSize = batchSize
        self.N_STATES = N_STATES
        self.train = train

    def run(self):

        print("开始线程：state {0}".format(self.threadID))
        step = train_with_play(self.batchSize,self.N_STATES,RL=self.train,mode=self.mode)
        print("退出线程：state {0}".format(self.threadID))
        return step



if __name__ == "__main__":
    """
    threadLFA = TrainThread("Mode -- {0} Thread".format("LFA"), "LFA")
    threadtable = TrainThread("Mode --  {0} Thread".format("table"), "table")

    threadLFA.start()
    threadtable.start()

    threadLFA.join()
    threadtable.join()

    print("Train is done...")
    """

    #train("priority")

    """Setting"""
    N_STATES = 40 # the length of the 1 dimensional world, #states + terminate
    ACTIONS = [1, 2]  # available actions 1:right, 2:wrong
    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 1 - (1 / (N_STATES - 1))  # discount factor
    MAX_EPISODES = 100  # maximum episodes
    batchSize = 20
    max_epoch = 600

    """
    Load the memory
    """
    #memory = np.load("/Users/roy/Documents/GitHub/MyAI/Run/blind_cliffwalk_experience/memory_25.npy")
    """
       test2 = linear_Q(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                                  discountFactor=GAMMA,
                                  Max_episode=MAX_EPISODES)
        #step2 = train_with_play(batchSize, N_STATES,RL= test2,mode="l")
    """

    """ The experimental of effect of Memory size and minibatch size """
    batchSize1 = 40
    memorysize1 = 25
    test = priority_train(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                          discountFactor=GAMMA,
                          Max_episode=MAX_EPISODES,
                          memorySize=memorysize1)
    batchSize2 = 40
    memorysize2 = 50
    test2 = priority_train(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                          discountFactor=GAMMA,
                          Max_episode=MAX_EPISODES,
                          memorySize=memorysize2)

    batchSize3 = 40
    memorysize3 = 75
    test3 = priority_train(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                           discountFactor=GAMMA,
                           Max_episode=MAX_EPISODES,
                           memorySize=memorysize3)

    batchSize4 = 40
    memorysize4 = 100
    test4 = priority_train(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                           discountFactor=GAMMA,
                           Max_episode=MAX_EPISODES,
                           memorySize=memorysize4)

    batchSize5 = 40
    memorysize5 = 125
    test5 = priority_train(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                           discountFactor=GAMMA,
                           Max_episode=MAX_EPISODES,
                           memorySize=memorysize5)

    batchSize6 = 40
    memorysize6 = 150
    test6 = priority_train(numState=N_STATES, ActionSet=ACTIONS, greedy=EPSILON, learnRate=ALPHA,
                           discountFactor=GAMMA,
                           Max_episode=MAX_EPISODES,
                           memorySize=memorysize6)


    step, awc = train_with_play(batchSize, N_STATES,RL= test,mode="p")
    step2, awc2 = train_with_play(batchSize, N_STATES,RL= test2,mode="p")
    step3, awc3 = train_with_play(batchSize, N_STATES, RL=test3, mode="p")
    step4, awc4 = train_with_play(batchSize, N_STATES, RL=test4, mode="p")
    step5, awc5 = train_with_play(batchSize, N_STATES, RL=test5, mode="p")
    step6, awc6 = train_with_play(batchSize, N_STATES, RL=test6, mode="p")


    plt.figure(1)

    plt.plot(step[0, :], step[1, :] - step[1, 0], c='r', label='memory size: 25 ')
    plt.plot(step2[0, :], step2[1, :] - step2[1, 0], c='b', label='memory size: 50 ')
    plt.plot(step3[0, :], step3[1, :] - step3[1, 0], c='g', label='memory size: 75 ')
    plt.plot(step4[0, :], step4[1, :] - step4[1, 0], c='c', label='memory size: 100 ')
    plt.plot(step5[0, :], step5[1, :] - step5[1, 0], c='k', label='memory size: 125 ')
    plt.plot(step6[0, :], step6[1, :] - step6[1, 0], c='m', label='memory size: 150 ')

    plt.legend(loc='best')
    plt.ylabel('Training step increment')
    plt.xlabel('Episode')
    plt.grid()
    plt.show()

    plt.figure(2)

    plt.plot(step[0, :], step[1, :], c='r', label='memory size: 25 ')
    plt.plot(step2[0, :], step2[1, :], c='b', label='memory size: 50 ')
    plt.plot(step3[0, :], step3[1, :], c='g', label='memory size: 75 ')
    plt.plot(step4[0, :], step4[1, :], c='c', label='memory size: 100 ')
    plt.plot(step5[0, :], step5[1, :], c='k', label='memory size: 125 ')
    plt.plot(step6[0, :], step6[1, :], c='m', label='memory size: 150 ')

    plt.legend(loc='best')
    plt.ylabel('Total training time')
    plt.xlabel('Episode')
    plt.grid()
    plt.show()

    plt.figure(3)

    plt.plot(awc[0, :], awc[1, :], c='r', label='memory size: 25 ')
    plt.plot(awc2[0, :], awc2[1, :], c='b', label='memory size: 50 ')
    plt.plot(awc3[0, :], awc3[1, :], c='g', label='memory size: 75 ')
    plt.plot(awc4[0, :], awc4[1, :], c='c', label='memory size: 100 ')
    plt.plot(awc5[0, :], awc5[1, :], c='k', label='memory size: 125 ')
    plt.plot(awc6[0, :], awc6[1, :], c='m', label='memory size: 150 ')

    plt.legend(loc='best')
    plt.ylabel('Mean of accumulate weight-change')
    plt.xlabel('Learning steps')
    plt.grid()
    plt.show()


    """
    plt.figure(1)
    plt.plot(step2[0, :], step2[1, :] - step2[1, 0], c='b', label='Uniform linear FA')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()
    """