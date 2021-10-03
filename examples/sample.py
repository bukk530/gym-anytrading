import gym
import gym_manytrading

import matplotlib.pyplot as plt

env = gym.make('sample-v0')

print(env.action_space)

state = env.reset()
while True:
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    if done:
        print("info", info)
        break

plt.figure(figsize=(15, 6))
plt.cla()
env.render_all()
plt.show()