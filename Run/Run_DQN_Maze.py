from DQN import DQN
from training_env import Maze
import numpy as np


def run_maze(env, brain, no_op_max=300, episode_max=10000000, iterStep=5):
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
            if reward == 1:
                for i in range(50):
                    brain.store_exp(e)
            elif reward == -1:
                for ii in range(10):
                    brain.store_exp(e)
            else:
                brain.store_exp(e)

            if (step > no_op_max) and (step % iterStep == 0):
                print("\n=================================================================\n")
                print("--> Start to Learn...")
                brain.learn()
                #time.sleep(2)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if is_done:
                #print("--> Number of Reward=1: {0}".format(len(np.where(brain.memory == 1)[0])))
                break
            step += 1

    # end of game
    print('game over')
    brain.save_the_model(brain.sess)  # Save the model
    env.destroy()

if __name__ == '__main__':
    # maze game
    env = Maze.Maze_env()
    n_action = len(env.actionSet)
    n_features = env.n_features
    RL = DQN.DQN_st(n_action, n_features,minibatch_size=100)
    env.after(100, run_maze(env=env,brain=RL))
    env.mainloop()