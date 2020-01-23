import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt


class LinearTrackingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    LOCKED = 3
    SEARCH = 2
    TARGET = 1

    def __init__(self, grid_size=5, time_delta=0.1):
        self.time_delta = time_delta
        self.grid_size = grid_size
        self.position = np.zeros(2)
        self.cell = np.zeros(2, dtype=int)
        self.velocity = np.zeros(2)
        self.time_delta = time_delta
        self.cell_size = 1 / grid_size

        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        self._n_obs = 0

        self.action_space = spaces.Discrete(grid_size ** 2)
        self.beam = np.zeros(2, dtype=int)

        self.image = None
        self.ax = None
        self.visualization = np.zeros((grid_size, grid_size), dtype=int)

    def step(self, action):
        self._update_beam(action)

        if self._is_locked():
            reward = 1
        else:
            reward = -1

        self._update_visualization()

        self._move_target()
        self._update_observation()

        if self._target_out():
            done = True
        else:
            done = False

        self._n_obs += 1
        return self.obs, reward, done, {}

    def reset(self):
        self._reset_target()
        self._update_observation()
        self._n_obs = 0 
        return self.obs

    def render(self, mode='human', close=False):
        if self.image == None:
            self.ax = plt.subplot(111)
            self.image = plt.imshow(self.visualization.T, vmin=0, vmax=3, cmap='jet')
        else:
            self.image.set_data(self.visualization.T)
        self.ax.set_title(str(self._n_obs))
        plt.show(block=False)

    def _move_target(self):
        self.position = self.position  + self.time_delta * self.velocity
        self._update_cell()
    
    def _reset_target(self):
        self.position = np.random.rand(2)
        self.velocity = np.random.rand(2) * 2 - 1
        self.velocity = 0.5 * self.velocity / (np.linalg.norm(self.velocity))
        self._update_cell()

    def _update_cell(self):
        self.cell = np.floor(self.position / self.cell_size).astype(dtype=int)

    def _update_visualization(self):
        self.visualization.fill(0)
        if self._is_locked():
            self.visualization[tuple(self.beam)] = self.LOCKED
        else:
            self.visualization[tuple(self.beam)] = self.SEARCH
            self.visualization[tuple(self.cell)] = self.TARGET

    def _update_beam(self, action):
        self.beam[0] = action % self.grid_size
        self.beam[1] = action // self.grid_size

    def _target_out(self):
        if np.any(self.position >= 1) or np.any(self.position < 0):
            return True
        else:
            return False

    def _is_locked(self):
        if np.all(self.cell == self.beam):
            return True
        else:
            return False
        
    def _update_observation(self):
        self.obs[:2] = self.position
        self.obs[2:] = self.velocity

