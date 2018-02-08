from DQN import DQN
from training_env import Maze
import numpy as np
import time

def run_maze(env, brain, no_op_max=300, episode_max=1000, iterStep=5):
    step = 0
    for episode in range(episode_max):
        # initial observation
        observation = env.reset_exp()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = brain.choose_action(observation)

            observation_ = env.update_env(action)
            # RL take action and get next Robservation and reward
            reward, is_done, observation_= env.feedback()

            # Experience Transition
            e = np.hstack((observation, [action, reward], observation_))
            brain.store_exp(e)

            if (step > no_op_max) and (step % iterStep == 0):
                print("--> Current Step: {0}".format(step))
                print("--> Start to Learn...")
                brain.learn()
                #time.sleep(2)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if is_done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()

if __name__ == '__main__':
    # maze game
    env = Maze.Maze_env()
    n_action = len(env.actionSet)
    n_features = env.n_features
    RL = DQN.DQN_st(n_action, n_features,minibatch_size=100)
    env.after(100, run_maze(env=env,brain=RL))
    env.mainloop()
    RL.plot_cost()