from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np

env = gym_super_mario_bros.make('SuperMarioBros-v0')

# Loads generated level in numpy array format and processes array to fit size constraints
array = np.load('new_level.npy')
array = np.delete(array, 11, 0)

# Maps the GAN symbols to corresponding emulator symbols 
my_dict = {0: 0x54, 1: 0x51, 2: 0, 3: 0xc1, 10: 0xc0, 4: 0, 5:0, 6:0x12, 7:0x13, 8:0x14, 9:0x15, 11:0, 12:0}
array = np.vectorize(my_dict.get)(array)

np.save('data.npy', array)
array_data = np.load('data.npy')

# Sets custom level in gym environment and runs
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.set_custom_level(array_data)
done = True
for step in range(8000): # 5000
    if done:
        state = env.reset()
    state, reward, done, _, info = env.step(env.action_space.sample())
    env.render()

env.close()