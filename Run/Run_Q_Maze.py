from Q_learning.Q_Brain_Maze import Q_Brain
from training_env.Maze import Maze_env

if __name__ == '__main__':

    env = Maze_env()
    actionSet = env.actionSet

    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 0.9  # discount factor
    MAX_EPISODES = 100  # maximum episodes

    """ create the Q brain """
    Q_brain = Q_Brain(actionSet,ALPHA,GAMMA,EPSILON,MAX_EPISODES)

    env.after(100, Q_brain.training(env=env))
    env.mainloop()