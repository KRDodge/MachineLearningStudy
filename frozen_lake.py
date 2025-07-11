import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.registration import register
import random


def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)

register(id='FrozenLake-v3',
         entry_point='gym.envs.toy_text:FrozenLakeEnv',
         kwargs = {'map_name':'4x4', 'is_slippery':False})

env = gym.make('FrozenLake-v3')
Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 2000
dis = .99

rList = []
for i in range(num_episodes):
    state = env.reset()[0]
    rAll = 0
    done = False

    while not done:
        action = np.argmax((Q[state, :]) + np.random.randn(1, env.action_space.n) / (i + 1))
        new_state, reward, terminated, truncated, _ = env.step(action)

        Q[state, action] = reward + dis * np.max(Q[new_state, :])
        done = terminated or truncated
        rAll += reward
        state = new_state

    rList.append(rAll)


print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(num_episodes), rList, color = "blue")
plt.show()