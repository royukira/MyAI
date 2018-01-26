from SARSA import sarsa_Brain
from training_env.Maze import Maze_env

if __name__ == '__main__':

    env = Maze_env()
    actionSet = env.actionSet

    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 0.9  # discount factor
    MAX_EPISODES = 100  # maximum episodes

    """ create the Q brain """
    s_brain = sarsa_Brain.sarsa(actionSet,ALPHA,GAMMA,EPSILON,MAX_EPISODES)

    env.after(100, s_brain.training(env=env))
    env.mainloop()