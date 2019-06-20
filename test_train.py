import math
import time
import sys
from collections import deque

import numpy as np

from agents import DDPG_Agent, PolicySearch_Agent
from task import Task

def train(agent:object, task:Task, num_episodes:int) -> tuple:
    """
    Train agent
    """

    # initialize total rewards
    total_rewards:deque = deque(maxlen=100000)

    for i_episode in range(1, num_episodes+1):
        # start a new episode
        state:np.ndarray = agent.reset_episode()

        #initialize episode rewards
        episode_rewards:float = 0.

        while True:
            action:np.ndarray = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            episode_rewards += reward

            if done:

                if hasattr(agent, 'memory'):
                    memory_rewards:list = [i.reward for i in agent.memory.memory]
                    total_rewards.append(episode_rewards)
                    print("\r Episode {}/{} | Average rewards {}".format(i_episode, num_episodes, np.mean(memory_rewards[:100])), end="")

                elif hasattr(agent, 'score'):
                    print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                          i_episode, agent.score, agent.best_score, agent.noise_scale), end="")  # [debug]

                sys.stdout.flush()

                break

    # print final position
    fin_pos:np.ndarray = task.sim.pose[:3]
    target_pos:np.ndarray = task.target_pos
    print("\n", agent.__class__.__name__, fin_pos, target_pos)
    sys.stdout.flush()

    return (total_rewards, fin_pos)

if __name__ == '__main__':

    num_episodes = 500

    setup = {
        "init_pose": np.array([0., 0., 10., 0., 0., 0.]),
        "init_velocities": np.array([0., 0., 0.]),
        "init_angle_velocities": np.array([0., 0., 0.]),
        "target_pos": np.array([5., 5., 20.]), # 0,0,10 was default
        "runtime": 5.
    }

    my_task:Task = Task(**setup)
    ddpg_agent:DDPG_Agent = DDPG_Agent(my_task)
    ps_agent:PolicySearch_Agent = PolicySearch_Agent(my_task)

    total_rewards:deque = train(ddpg_agent, my_task, num_episodes)
    # train(ps_agent, my_task, num_episodes)
